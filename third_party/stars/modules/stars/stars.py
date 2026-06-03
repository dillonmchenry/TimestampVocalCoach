from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.commons.hparams import hparams
from utils.commons.gpu_mem_track import MemTracker
from modules.commons.layers import Embedding, LayerNorm
from modules.commons.transformer import SinusoidalPositionalEmbedding
from modules.commons.conv import ResidualBlock, ConvBlocks
from modules.stars.utils import LocalStyleAdaptor, ProsodyAligner, get_ph_word_bd
import time

def regulate_boundary(bd_logits, threshold, min_gap=18, ref_bd=None, ref_bd_min_gap=8, non_padding=None):
    # this doesn't preserve gradient
    device = bd_logits.device
    bd_logits = torch.sigmoid(bd_logits).data.cpu()
    bd = (bd_logits > threshold).long()
    bd_res = torch.zeros_like(bd).long()
    for i in range(bd.shape[0]):
        bd_i = bd[i]
        last_bd_idx = -1
        start = -1
        for j in range(bd_i.shape[0]):
            if bd_i[j] == 1:
                if 0 <= start < j:
                    continue
                elif start < 0:
                    start = j
            else:
                if 0 <= start < j:
                    if j - 1 > start:
                        bd_idx = start + int(torch.argmax(bd_logits[i, start: j]).item())
                    else:
                        bd_idx = start
                    if bd_idx - last_bd_idx < min_gap and last_bd_idx > 0:
                        bd_idx = round((bd_idx + last_bd_idx) / 2)
                        bd_res[i, last_bd_idx] = 0
                    bd_res[i, bd_idx] = 1
                    last_bd_idx = bd_idx
                    start = -1

    # assert ref_bd_min_gap <= min_gap // 2
    if ref_bd is not None and ref_bd_min_gap > 0:
        ref = ref_bd.data.cpu()
        for i in range(bd_res.shape[0]):
            ref_bd_i = ref[i]
            ref_bd_i_js = []
            for j in range(ref_bd_i.shape[0]):
                if ref_bd_i[j] == 1:
                    ref_bd_i_js.append(j)
                    seg_sum = torch.sum(bd_res[i, max(0, j - ref_bd_min_gap): j + ref_bd_min_gap])
                    if seg_sum == 0:
                        bd_res[i, j] = 1
                    elif seg_sum == 1 and bd_res[i, j] != 1:
                        bd_res[i, max(0, j - ref_bd_min_gap): j + ref_bd_min_gap] = \
                            ref_bd_i[max(0, j - ref_bd_min_gap): j + ref_bd_min_gap]
                    elif seg_sum > 1:
                        for k in range(1, ref_bd_min_gap+1):
                            if bd_res[i, max(0, j - k)] == 1 and ref_bd_i[max(0, j - k)] != 1:
                                bd_res[i, max(0, j - k)] = 0
                                break
                            if bd_res[i, min(bd_res.shape[1] - 1, j + k)] == 1 and ref_bd_i[min(bd_res.shape[1] - 1, j + k)] != 1:
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] = 0
                                break
                        bd_res[i, j] = 1
            # final check
            assert torch.sum(bd_res[i, ref_bd_i_js]) == len(ref_bd_i_js), \
                f"{torch.sum(bd_res[i, ref_bd_i_js])} {len(ref_bd_i_js)}"

    bd_res = bd_res.to(device)
    bd_res[:, 0] = 0
    if non_padding is not None:
        for i in range(bd_res.shape[0]):
            bd_res[i, int(sum(non_padding[i]).item()) - 1:] = 0
    else:
        bd_res[:, -1] = 0

    return bd_res

class PitchDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get('note_bd_temperature', 1.0))

        # note prediction
        self.pitch_attn_num_head = hparams.get('pitch_attn_num_head', 1)
        self.multihead_dot_attn = nn.Linear(hidden_size, self.pitch_attn_num_head)
        self.post = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                                post_net_kernel=3, act_type='leakyrelu')
        self.pitch_out = nn.Linear(hidden_size, hparams.get('note_num', 100) + 4)
        self.note_num = hparams.get('note_num', 100)
        self.note_start = hparams.get('note_start', 30)
        self.pitch_temperature = max(1e-7, hparams.get('note_pitch_temperature', 1.0))

    def forward(self, feat, note_bd, train=True):
        bsz, T, _ = feat.shape
        attn = torch.sigmoid(self.multihead_dot_attn(feat))  # [B, T, C] -> [B, T, num_head]
        attn = F.dropout(attn, self.dropout, train)
        attn_feat = feat.unsqueeze(3) * attn.unsqueeze(2)  # [B, T, C, 1] x [B, T, 1, num_head] -> [B, T, C, num_head]
        attn_feat = torch.mean(attn_feat, dim=-1)  # [B, T, C, num_head] -> [B, T, C]
        mel2note = torch.cumsum(note_bd, 1)
        note_length = torch.max(torch.sum(note_bd, dim=1)).item() + 1  # max length
        note_lengths = torch.sum(note_bd, dim=1) + 1  # [B]

        attn = torch.mean(attn, dim=-1, keepdim=True)  # [B, T, num_head] -> [B, T, 1]
        denom = mel2note.new_zeros(bsz, note_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2note, src=attn.squeeze(-1)
        )  # [B, T] -> [B, note_length] count the note frames of each note (with padding excluded)
        frame2note = mel2note.unsqueeze(-1).repeat(1, 1, self.hidden_size)  # [B, T] -> [B, T, C], with padding included
        note_aggregate = frame2note.new_zeros(bsz, note_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2note, src=attn_feat
        )  # [B, T, C] -> [B, note_length, C]
        note_aggregate = note_aggregate / (denom.unsqueeze(-1) + 1e-5)
        note_aggregate = F.dropout(note_aggregate, self.dropout, train)
        note_logits = self.post(note_aggregate)
        note_logits = self.pitch_out(note_logits) / self.pitch_temperature

        note_pred = torch.softmax(note_logits, dim=-1)  # [B, note_length, note_num]
        note_pred = torch.argmax(note_pred, dim=-1)  # [B, note_length]
        note_pred[note_pred > self.note_num] = 0
        note_pred[note_pred < self.note_start] = 0

        return note_lengths, note_logits, note_pred

class StylePredictor(nn.Module):
    def __init__(self, hparams):
        super(StylePredictor, self).__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.tech_norm = LayerNorm(hidden_size)
        self.lan_norm = LayerNorm(hidden_size)
        self.gen_norm = LayerNorm(hidden_size)
        self.emo_norm = LayerNorm(hidden_size)
        self.meth_norm = LayerNorm(hidden_size)
        self.pace_norm = LayerNorm(hidden_size)
        self.range_norm = LayerNorm(hidden_size)

        self.gen_head = nn.Linear(hidden_size, hparams.get('num_gender_tokens', 2))
        self.meth_head = nn.Linear(hidden_size, hparams.get('num_method_tokens', 2))
        self.tech_head = nn.Linear(hidden_size, hparams.get('num_technique_tokens', 10))
        self.lan_head = nn.Linear(hidden_size, hparams.get('num_language_tokens', 9))
        self.emo_head = nn.Linear(hidden_size, hparams.get('num_emotion_tokens', 4))
        self.pace_head = nn.Linear(hidden_size, hparams.get('num_pace_tokens', 3))
        self.range_head = nn.Linear(hidden_size, hparams.get('num_range_tokens', 3))
        self.reset_parameters()
    
    def forward(self, feat, ret, train):
        tech_token = feat[:, 0]
        lan_token = feat[:, 1]
        gen_tokens = feat[:, 2]
        emo_token = feat[:, 3]
        meth_token = feat[:, 4]
        pace_tokens = feat[:, 5]
        range_token = feat[:, 6]

        tech_token = self.tech_norm(tech_token)
        lan_token = self.lan_norm(lan_token)
        gen_tokens = self.gen_norm(gen_tokens)
        emo_token = self.emo_norm(emo_token)
        meth_token = self.meth_norm(meth_token)
        pace_tokens = self.pace_norm(pace_tokens)
        range_token = self.range_norm(range_token)
        
        ret['technique_logits'] = technique_logits = self.tech_head(F.dropout(tech_token, self.dropout, train))
        ret['language_logits'] = language_logits = self.lan_head(F.dropout(lan_token, self.dropout, train))
        ret['gender_logits'] = gender_logits = self.gen_head(F.dropout(gen_tokens, self.dropout, train))
        ret['emotion_logits'] = emotion_logits = self.emo_head(F.dropout(emo_token, self.dropout, train))
        ret['method_logits'] = method_logits = self.meth_head(F.dropout(meth_token, self.dropout, train))
        ret['pace_logits'] = pace_logits = self.pace_head(F.dropout(pace_tokens, self.dropout, train))
        ret['range_logits'] = range_logits = self.range_head(F.dropout(range_token, self.dropout, train))
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.tech_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.lan_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.gen_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.emo_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.meth_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.pace_head.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.range_head.weight, mode='fan_in')
        nn.init.constant_(self.tech_head.bias, 0.0)
        nn.init.constant_(self.lan_head.bias, 0.0)
        nn.init.constant_(self.gen_head.bias, 0.0)
        nn.init.constant_(self.emo_head.bias, 0.0)
        nn.init.constant_(self.meth_head.bias, 0.0)
        nn.init.constant_(self.pace_head.bias, 0.0)
        nn.init.constant_(self.range_head.bias, 0.0)

class TechniquePredictor(nn.Module):
    def __init__(self, hparams):
        super(TechniquePredictor, self).__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.tech_post = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                layers_in_block=1, c_multiple=1, dropout=self.dropout, num_layers=1,
                                post_net_kernel=3, act_type='swish')
        self.tech_attn_num_head = hparams.get('tech_attn_num_head', 1)
        self.multihead_tech_attn = nn.Linear(hidden_size, self.tech_attn_num_head)
        self.binary_tech_out = nn.Linear(hidden_size, hparams.get('binary_tech_num', 5))
        self.tech_temperature = max(1e-7, hparams.get('tech_temperature', 1.0))
        self.tech_threshold = hparams.get('tech_threshold', 0.5)
        self.reset_parameters()

    def forward(self, feat, ret, ph_bd, train):
        bsz = feat.shape[0]
        attn = torch.sigmoid(self.multihead_tech_attn(feat))     # [B, T, C] -> [B, T, num_head]
        attn = F.dropout(attn, self.dropout, train)
        attn_feat = feat.unsqueeze(3) * attn.unsqueeze(2)   # [B, T, C, 1] x [B, T, 1, num_head] -> [B, T, C, num_head]
        attn_feat = torch.mean(attn_feat, dim=-1)   # [B, T, C, num_head] -> [B, T, C]
        
        mel2ph = torch.cumsum(ph_bd, 1)
        ph_length = torch.max(torch.sum(ph_bd, dim=1)).item() + 1  # [B]
        attn = torch.mean(attn, dim=-1, keepdim=True)   # [B, T, num_head] -> [B, T, 1]
        denom = mel2ph.new_zeros(bsz, ph_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2ph, src=attn.squeeze(-1)
        )  # [B, T] -> [B, ph_length] count the note frames of each note (with padding excluded)
        frame2ph = mel2ph.unsqueeze(-1).repeat(1, 1, self.hidden_size)  # [B, T] -> [B, T, C], with padding included
        ph_aggregate = frame2ph.new_zeros(bsz, ph_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2ph, src=attn_feat
        )  # [B, T, C] -> [B, ph_length, C]
        ph_aggregate = ph_aggregate / (denom.unsqueeze(-1) + 1e-5)
        ph_aggregate = F.dropout(ph_aggregate, self.dropout, train)
        tech_logits = self.tech_post(ph_aggregate)
        
        tech_logits = self.binary_tech_out(tech_logits) / self.tech_temperature
        ret['tech_logits'] = tech_logits    # [B, ph_length, note_num]
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.multihead_tech_attn.weight, mode='fan_in')
        nn.init.constant_(self.multihead_tech_attn.bias, 0.0)
        nn.init.kaiming_normal_(self.binary_tech_out.weight, mode='fan_in') 
        nn.init.constant_(self.binary_tech_out.bias, 0.0)

class PhFramePredictor(nn.Module):
    def __init__(self, hparams):
        super(PhFramePredictor, self).__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.ph_head = nn.Linear(hidden_size, hparams.get('ph_num', 58) + 3)
        self.ph_bd_temperature = max(1e-7, hparams.get('ph_bd_temperature', 1.0))
        self.reset_parameters()
    
    def forward(self, feat, ret, train):
        frame_logits = self.ph_head(F.dropout(feat, self.dropout, train)).squeeze(-1) 
        ph_bd_logits = frame_logits[:,:,0] / self.ph_bd_temperature
        ph_bd_logits = torch.clamp(ph_bd_logits, min=-16., max=16.)
        ph_frame_logits = frame_logits[:,:,1:] / self.ph_bd_temperature
        ret['ph_bd_logits'], ret['ph_frame_logits'] = ph_bd_logits, ph_frame_logits
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.ph_head.weight, mode='fan_in')
        nn.init.constant_(self.ph_head.bias, 0.0)

class NoteFramePredictor(nn.Module):
    def __init__(self, hparams):
        super(NoteFramePredictor, self).__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.note_head = nn.Linear(hidden_size, hparams.get('note_num', 100) + 4 + 1)
        self.note_bd_temperature = max(1e-7, hparams.get('note_bd_temperature', 1.0))
        self.note_temperature = max(1e-7, hparams.get('note_temperature', 1.0))
        self.note_bd_threshold = hparams.get('note_bd_threshold', 0.5)
        self.note_bd_min_gap = round(hparams.get('note_bd_min_gap', 100) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
        self.note_bd_ref_min_gap = round(hparams.get('note_bd_ref_min_gap', 50) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
        self.reset_parameters()
    
    def forward(self, feat, ret, word_bd, non_padding, train):
        # ph bd prediction
        frame_logits = self.note_head(F.dropout(feat, self.dropout, train)).squeeze(-1) 
        note_bd_logits = frame_logits[:,:,0] / self.note_bd_temperature
        note_bd_logits = torch.clamp(note_bd_logits, min=-16., max=16.)
        note_frame_logits = frame_logits[:,:,1:] / self.note_bd_temperature
        ret['note_bd_logits'], ret['note_frame_logits'] = note_bd_logits, note_frame_logits

        if not train:
            ret['note_bd_pred'] = regulate_boundary(note_bd_logits, self.note_bd_threshold, self.note_bd_min_gap, word_bd, self.note_bd_ref_min_gap, non_padding)
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.note_head.weight, mode='fan_in')
        nn.init.constant_(self.note_head.bias, 0.0)

class STARS(nn.Module):
    def __init__(self, hparams):
        super(STARS, self).__init__()
        self.hparams = deepcopy(hparams)
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        self.padding_idx = 0
        self.mel_proj = nn.Conv1d(hparams['use_mel_bins'], hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                                        layers_in_block=2, c_multiple=1, dropout=self.dropout, num_layers=1,
                                        post_net_kernel=3, act_type='leakyrelu')
        self.pitch_embed = Embedding(300, hidden_size, 0, 'kaiming')
        self.uv_embed = Embedding(3, hidden_size, 0, 'kaiming')

        self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size, self.padding_idx, init_size=hparams['max_frames'],
            )
        # build prosody extractor
        ## frame level
        self.prosody_extractor_utter = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx, self.dropout, num_layers=hparams.get('conformer_layers', 1))
        self.l1_utter = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.ph_frame_predictor = PhFramePredictor(hparams)

        ## word level
        self.prosody_extractor_word = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx, self.dropout, num_layers=hparams.get('conformer_layers', 1))
        self.l1_word = nn.Linear(self.hidden_size * 2, self.hidden_size)

        ## phoneme level
        self.prosody_extractor_ph = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx, self.dropout, num_layers=hparams.get('conformer_layers', 1))
        self.l1_ph = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.note_frame_predictor = NoteFramePredictor(hparams)
        self.tech_predictor = TechniquePredictor(hparams)

        ## Note level
        self.prosody_extractor_note = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx, self.dropout, num_layers=hparams.get('conformer_layers', 1))
        self.l1_note = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.pitch_decoder = PitchDecoder(hparams)
        
        ## sentence level
        self.prosody_extractor_sentence = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx, self.dropout, num_layers=1)
        self.l1_sentence = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.align_sentence = ProsodyAligner(num_layers=2)

        self.num_cls_tokens = hparams.get('num_cls_tokens', 16)
        self.cls_tokens = nn.Parameter(torch.zeros(1, self.num_cls_tokens, hidden_size))
        self.style_predict = StylePredictor(hparams)

    def forward(self, mel=None, pitch=None, uv=None, mel_nonpadding=None, ph_nonpadding=None, word_nonpadding=None, note_nonpadding=None, mel2ph=None, mel2word=None, mel2note=None, word_bd=None, ph_bd=None, note_bd=None, ph=None, ph_lengths=None, mel_lengths=None, ph2words=None, train=True, global_steps=0):
        # mel (B, T, C) pitch (B, T) mel_nonpadding (B, T)
        ret = {}
        mel_embed = self.mel_proj(mel.transpose(1, 2)).transpose(1, 2)
        mel_embed = self.mel_encoder(mel_embed)
        mel_embed = mel_embed * mel_nonpadding.unsqueeze(-1)
        pitch_embed = self.pitch_embed(pitch) + self.uv_embed(uv)  # [B, T, C]
        pitch_embed = pitch_embed * mel_nonpadding.unsqueeze(-1)
        mel_embed = mel_embed + pitch_embed

        # add prosody VQ
        feat = self.get_prosody_utter(mel_embed, ret, mel_nonpadding, not train, global_steps)
        self.ph_frame_predictor(feat, ret, train=train)
        
        gt_ph_bd = ph_bd
        if not train:
            mel2ph, ph_bd, mel2word, word_bd, ret['ph_of_list'], ret['word_of_list'], ret['dp_matrix_list'] = get_ph_word_bd(ret, ph_lengths, mel_lengths, ph, ph2words)
        ret['word_bd'] = word_bd
        prosody_ph_mel = self.get_prosody_ph(mel_embed, ret, mel2ph, not train, global_steps)
        prosody_word_mel = self.get_prosody_word(mel_embed, ret, mel2word, not train, global_steps)
        feat = feat + prosody_ph_mel + prosody_word_mel
        self.note_frame_predictor(feat, ret, word_bd=None, non_padding=mel_nonpadding, train=train)
        if not train and global_steps > hparams['vq_note_start']:
            note_bd = ret['note_bd_pred']
            mel2note = torch.cumsum(note_bd, 1)
        prosody_word_mel = self.get_prosody_note(mel_embed, ret, mel2note, not train, global_steps)
        feat = feat + prosody_word_mel
        ret['note_lengths'], ret['note_logits'], ret['note_pred'] = self.pitch_decoder(feat, note_bd, train)
        
        feat = self.get_prosody_sentence(mel_embed, feat, mel_nonpadding, ret, not train, global_steps)
        if gt_ph_bd != None:
            self.tech_predictor(feat, ret, gt_ph_bd, train)
        else:
            self.tech_predictor(feat, ret, ph_bd, train)

        return ret

    def expand_states(self, h, mel2ph):
        h = F.pad(h, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2ph_)  # [B, T, H]
        return h

    def get_prosody_ph(self, mel_embed, ret, mel2ph, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_ph_start']:
            prosody_embedding, loss, ppl = self.prosody_extractor_ph(mel_embed, mel2ph, no_vq=False)
            ret['vq_loss_ph'] = loss
            ret['ppl_ph'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_ph(mel_embed, mel2ph, no_vq=True)
        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_ph(torch.cat([prosody_embedding, positions], dim=-1))
        prosody_embedding = self.expand_states(prosody_embedding, mel2ph)
        
        return prosody_embedding

    def get_prosody_word(self, mel_embed, ret, mel2word, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_word_start']:
            prosody_embedding, loss, ppl = self.prosody_extractor_word(mel_embed, mel2word, no_vq=False)
            ret['vq_loss_word'] = loss
            ret['ppl_word'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_word(mel_embed, mel2word, no_vq=True)

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_word(torch.cat([prosody_embedding, positions], dim=-1))
        prosody_embedding = self.expand_states(prosody_embedding, mel2word)

        return prosody_embedding
    
    def get_prosody_note(self, mel_embed, ret, mel2note, infer=False, global_steps=0):
        # get VQ prosody
        if global_steps > hparams['vq_note_start']:
            prosody_embedding, loss, ppl = self.prosody_extractor_note(mel_embed, mel2note, no_vq=False)
            ret['vq_loss_note'] = loss
            ret['ppl_note'] = ppl
        else:
            prosody_embedding = self.prosody_extractor_note(mel_embed, mel2note, no_vq=True)

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_note(torch.cat([prosody_embedding, positions], dim=-1))
        prosody_embedding = self.expand_states(prosody_embedding, mel2note)
        
        return prosody_embedding

    def get_prosody_utter(self, mel_embed, ret, mel_nonpadding, infer=False, global_steps=0):
        # get VQ prosody
        prosody_embedding = self.prosody_extractor_utter(mel_embed, no_vq=True)
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1_utter(torch.cat([prosody_embedding, positions], dim=-1))        

        return prosody_embedding

    def get_prosody_sentence(self, mel_embed, feat, mel_nonpadding, ret, infer=False, global_steps=0):
        bsz = mel_embed.shape[0]
        prosody_embedding = self.prosody_extractor_sentence(mel_embed, no_vq=True)
        prosody_embedding = prosody_embedding.mean(dim=1)
        output = feat + prosody_embedding.unsqueeze(1)
        
        cls_tokens = self.cls_tokens.expand(bsz, -1, -1)
        mel_padding_mask = (1 - mel_nonpadding).to(torch.bool)
        style_features, _, attn_emo = self.align_sentence(cls_tokens.transpose(0, 1), output.transpose(0, 1), None, mel_padding_mask)
        self.style_predict(style_features.transpose(0, 1), ret, not infer)
        return output
