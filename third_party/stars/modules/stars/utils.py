from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.commons.hparams import hparams
from utils.commons.gpu_mem_track import MemTracker
from modules.commons.layers import Embedding, LayerNorm
from modules.commons.transformer import FFTBlocks, DecSALayer
from modules.commons.conv import ResidualBlock, ConvBlocks
from modules.commons.conformer.conformer import ConformerLayers, ConformerLayersMOE
from modules.stars.unet import Unet
from scipy.cluster.vq import kmeans2
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d, pad_or_cut_xd

def group_hidden_by_segs(h, seg_ids, max_len):
    """

    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs, cnt_gby_segs
def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )
class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5,
                 print_vq_prob=False):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.print_vq_prob = print_vq_prob
        self.register_buffer('data_initialized', torch.zeros(1))
        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # [B*T_mel, N_vq]
        indices = torch.argmin(distances.float(), dim=-1)  # [B*T_mel]
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()
        if self.training and self.data_initialized.item() == 0:
            print('| running kmeans in VQVAE')  # data driven initialization for the embeddings
            x_flat = x.detach().reshape(-1, D)
            rp = torch.randperm(x_flat.size(0))
            kd = kmeans2(x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit='points')
            self.embedding.copy_(torch.from_numpy(kd[0]))
            x_flat, quantized, indices = self.encode(x)
            encodings = F.one_hot(indices, M).float()
            self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
            self.ema_count.copy_(torch.sum(encodings, dim=0))

        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)

        if self.training and self.data_initialized.item() != 0:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        self.data_initialized.fill_(1)

        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction='none')
        nonpadding = (x.abs().sum(-1) > 0).float()
        e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.print_vq_prob:
            print("| VQ code avg_probs: ", avg_probs)
        return quantized, loss, indices, perplexity

class CMUEncoder(nn.Module):
    def __init__(self, hparams, num_layers):
        super().__init__()
        self.hidden_size = hidden_size = hparams['hidden_size']
        self.dropout = hparams.get('dropout', 0.0)
        if hparams.get('updown_rates', None) is not None:
            updown_rates = [int(i) for i in hparams.get('updown_rates', None).split('-')]
        if hparams.get('channel_multiples', None) is not None:
            channel_multiples = [float(i) for i in hparams.get('channel_multiples', None).split('-')]
        assert len(updown_rates) == len(channel_multiples)
        if num_layers != None:
            self.num_layers = num_layers
        else:
            self.num_layers = hparams.get('bkb_layers', 2)
        if hparams.get('bkb_net', 'conformer') == 'conformer':
            if hparams.get('use_fremoe', True):
                mid_net = ConformerLayersMOE(
                    hidden_size, num_layers=self.num_layers, kernel_size=hparams.get('conformer_kernel', 9),
                    dropout=self.dropout, num_heads=4)
            else:
                mid_net = ConformerLayers(
                    hidden_size, num_layers=self.num_layers, kernel_size=hparams.get('conformer_kernel', 9),
                    dropout=self.dropout, num_heads=4)

            self.net = Unet(hidden_size, down_layers=len(updown_rates), up_layers=len(updown_rates), kernel_size=3,
                            updown_rates=updown_rates, channel_multiples=channel_multiples, dropout=0,
                            is_BTC=True, constant_channels=False, mid_net=mid_net,
                            use_skip_layer=hparams.get('unet_skip_layer', False))
        elif hparams.get('bkb_net', 'conformer') == 'conv':
            self.net = Unet(hidden_size, down_layers=len(updown_rates), mid_layers=hparams.get('bkb_layers', 12),
                            up_layers=len(updown_rates), kernel_size=3, updown_rates=updown_rates,
                            channel_multiples=channel_multiples, dropout=0, is_BTC=True,
                            constant_channels=False, mid_net=None, use_skip_layer=hparams.get('unet_skip_layer', False))
        
    def forward(self, x, padding_mask=None):
        return self.net(x)

class LocalStyleAdaptor(nn.Module):
    def __init__(self, hidden_size, num_vq_codes=64, padding_idx=0, dropout=0.0, num_layers=None):
        super(LocalStyleAdaptor, self).__init__()
        self.encoder = ConvBlocks(hidden_size, out_dims=hidden_size, dilations=None, kernel_size=3,
                        layers_in_block=1, c_multiple=1, dropout=dropout, num_layers=1,
                        post_net_kernel=3, act_type='leakyrelu')
        self.n_embed = num_vq_codes
        self.vqvae = VQEmbeddingEMA(self.n_embed, hidden_size, commitment_cost=hparams['lambda_commit'])
        self.cmuencoder = CMUEncoder(hparams, num_layers)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

    def forward(self, ref_mels, mel2ph=None, no_vq=False):
        """

        :param ref_mels: [B, T, C]
        :return: [B, 1, H]
        """
        ref_mels = self.cmuencoder(ref_mels)
        if mel2ph is not None:
            ref_ph, _ = group_hidden_by_segs(ref_mels, mel2ph, torch.max(mel2ph))
        else:
            ref_ph = ref_mels
        prosody = self.encoder(ref_ph)
        if no_vq:
            return prosody
        z, vq_loss, vq_tokens, ppl = self.vqvae(prosody)
        vq_loss = vq_loss.mean()
        return z, vq_loss, ppl

class CrossAttenLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttenLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, local_emotion, emotion_key_padding_mask=None):
        # src: (Tph, B, 256) local_emotion: (Temo, B, 256) emotion_key_padding_mask: (B, Temo)
        src2, attn_emo = self.multihead_attn(src, local_emotion, local_emotion, key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_emo

class ProsodyAligner(nn.Module):
    def __init__(self, num_layers, guided_sigma=0.3, guided_layers=None, norm=None):
        super(ProsodyAligner, self).__init__()
        self.layers = nn.ModuleList([CrossAttenLayer(d_model=hparams['hidden_size'], nhead=2) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

    def forward(self, src, local_emotion, src_key_padding_mask=None, emotion_key_padding_mask=None):
        output = src
        guided_loss = 0
        attn_emo_list = []
        for i, mod in enumerate(self.layers):
            # output: (Tph, B, 256), global_emotion: (1, B, 256), local_emotion: (Temo, B, 256) mask: None, src_key_padding_mask: (B, Tph),
            # emotion_key_padding_mask: (B, Temo)
            output, attn_emo = mod(output, local_emotion, emotion_key_padding_mask=emotion_key_padding_mask)
            attn_emo_list.append(attn_emo.unsqueeze(1))
            # attn_emo: (B, Tph, Temo) attn: (B, Tph, Tph)
            if i < self.guided_layers and src_key_padding_mask is not None:
                s_length = (~src_key_padding_mask).float().sum(-1) # B
                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                g_loss_emo = attn_emo * attn_w_emo  # N, L, S
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

        if self.norm is not None:
            output = self.norm(output)

        return output, guided_loss, attn_emo_list

def run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label, bd_log_prediction, not_bd_log_prediction):
    for j in range(1, cur_log_prediction.shape[0]):
        for k in range(cur_label.shape[0] * 2 + 1):
            if k == 0:
                # blank
                backtrace_dp_matrix[j][k] = k
                dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j] + not_bd_log_prediction[j]

            elif k == 1:
                if dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[0] - 2] + not_bd_log_prediction[j]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[0] - 2] + bd_log_prediction[j]

            elif k % 2 == 0:
                # blank
                if dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j] + not_bd_log_prediction[j]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_silence_prediction[j] + bd_log_prediction[j]

            else:
                if dp_matrix[j-1][k-2] + bd_log_prediction[j] >= dp_matrix[j-1][k-1] + bd_log_prediction[j] and dp_matrix[j-1][k-2] + bd_log_prediction[j] >= dp_matrix[j-1][k] + not_bd_log_prediction[j]:
                    # k-2 (last character) -> k
                    backtrace_dp_matrix[j][k] = k - 2
                    dp_matrix[j][k] = dp_matrix[j-1][k-2] + cur_log_prediction[j][cur_label[k // 2] - 2] + bd_log_prediction[j]

                elif dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    # k -> k
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[k // 2] - 2] + not_bd_log_prediction[j]
                else:
                    # k-1 -> k
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[k // 2] - 2] + bd_log_prediction[j]

    return dp_matrix, backtrace_dp_matrix


def perform_viterbi_bd(prediction, labels, ph_bd_logits, hop_size_second=0.02, ph2word=None):
    ph_bd_pred = torch.nn.functional.sigmoid(ph_bd_logits.float()).numpy()
    bd_prob_log = np.log(ph_bd_pred + 1e-6).astype("float32")
    not_bd_prob_log = np.log(1 - ph_bd_pred + 1e-6).astype("float32")

    ph_logs = F.log_softmax(prediction[:,1:], dim=-1)
    log_silence_prediction = ph_logs[:, 1]
    log_silence_prediction = torch.clip(log_silence_prediction, min=-1000)
    log_prediction = torch.clip(ph_logs[:, 2:], min=-1000)
    assert len(labels) == len(ph2word), f'the lengths of phs and ph2word is different'
    cur_ph2word = np.array([ph2word[j] for j in range(len(labels)) if labels[j] > 1 ])
    #remove 0(UNK), 1（SP）
    cur_label = np.array([labels[j] for j in range(len(labels)) if labels[j] > 1 ])
    
    # add SP
    dp_matrix = np.array([[-10000000.0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[0])])
    backtrace_dp_matrix = np.array([[0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[0])])

    cur_log_prediction = log_prediction.numpy()
    cur_log_silence_prediction = log_silence_prediction.numpy()
    dp_matrix[0][0] = cur_log_silence_prediction[0]
    dp_matrix[0][1] = cur_log_prediction[0][cur_label[0] - 2]

    dp_matrix, backtrace_dp_matrix = run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label, bd_prob_log, not_bd_prob_log)
    if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
        # Go backtrace
        # Get dp_matrix.shape[1] but dp_matrix is not numpy array XD
        correct_path = [len(dp_matrix[0]) - 1, ]
        cur_k = backtrace_dp_matrix[-1][-1]
        for j in range(len(dp_matrix)-2, -1, -1):
            correct_path.append(cur_k)
            cur_k = backtrace_dp_matrix[j][cur_k]
    else:
        correct_path = [len(dp_matrix[0]) - 2, ]
        cur_k = backtrace_dp_matrix[-1][-2]
        for j in range(len(dp_matrix)-2, -1, -1):
            correct_path.append(cur_k)
            cur_k = backtrace_dp_matrix[j][cur_k]
    correct_path.reverse()

    end_index = 0
    ph_of = []
    word_of = []
    cur_predicted_onset_offset = []
    for k in range(len(cur_label)):
        ph_id = cur_label[k]
        first_index = correct_path.index(k * 2 + 1)
        last_index = len(correct_path) - correct_path[::-1].index(k * 2 + 1) - 1
        cur_predicted_onset_offset.append([float(first_index) * hop_size_second, float(last_index + 1) * hop_size_second, ph_id])
        word_id = cur_ph2word[k]
        if first_index - end_index > 0.1 / hop_size_second and (len(ph_of) == 0 or ph_of[-1][-1] != word_id):
            ph_of.append([end_index, first_index, 1, word_id])
            word_of.append([end_index, first_index, -1])
            end_index = first_index
        ph_of.append([end_index, last_index + 1, cur_label[k], word_id])

        if len(word_of) == 0 or word_of[-1][-1] != word_id:
            word_of.append([end_index, last_index + 1, word_id])
        else:
            word_of[-1] = [word_of[-1][0], last_index + 1, word_id]
        end_index = last_index + 1

    if end_index != len(correct_path):
        ph_of.append([end_index, len(correct_path), 1, word_id + 1])
        word_of.append([end_index, len(correct_path), -1])

    mel2ph = torch.zeros(len(correct_path))
    mel2word = torch.zeros(len(correct_path))
    for ph_idx, ph in enumerate(ph_of):
        start_index = ph[0]
        end_index = ph[1]
        mel2ph[start_index:end_index] = ph_idx + 1

    for word_idx, word in enumerate(word_of):
        start_index = word[0]
        end_index = word[1]
        mel2word[start_index:end_index] = word_idx + 1

    ph_bd = torch.zeros_like(mel2ph)
    word_bd = torch.zeros_like(mel2word)
    ph_bd[1:] = (mel2ph[1:] - mel2ph[:-1] == 1).float()
    word_bd[1:] = (mel2word[1:] - mel2word[:-1] == 1).float()
    
    return mel2ph, ph_bd, mel2word, word_bd, ph_of, word_of, dp_matrix

def get_ph_word_bd(ret, ph_lengths, mel_lengths, phs, ph2words):
    bsz = ph_lengths.shape[0]
    mel2ph_list = []
    ph_bd_list = []
    mel2word_list = []
    word_bd_list = []
    ph_of_list = []
    word_of_list = []
    dp_matrix_list = []
    for idx in range(bsz):
        label = phs[idx].data.cpu()[:ph_lengths[idx]]
        ph2word = ph2words[idx].data.cpu()[:ph_lengths[idx]]
        ph_frame_logits = ret['ph_frame_logits'][idx].data.cpu()[:mel_lengths[idx]]
        ph_bd_logits = torch.sigmoid(ret['ph_bd_logits'])[idx].data.cpu()[:mel_lengths[idx]]
        mel2ph, ph_bd, mel2word, word_bd, ph_of, word_of, dp_matrix = perform_viterbi_bd(ph_frame_logits, label, ph_bd_logits, ph2word=ph2word)
        mel2ph_list.append(mel2ph)
        ph_bd_list.append(ph_bd)
        mel2word_list.append(mel2word)
        word_bd_list.append(word_bd)
        ph_of_list.append(ph_of)
        word_of_list.append(word_of)
        dp_matrix_list.append(dp_matrix)

    mel2ph = collate_1d_or_2d([s for s in mel2ph_list], 0).to(ret['ph_frame_logits'].device).long()
    ph_bd = collate_1d_or_2d([s for s in ph_bd_list], 0).to(ret['ph_frame_logits'].device).long()
    mel2word = collate_1d_or_2d([s for s in mel2word_list], 0).to(ret['ph_frame_logits'].device).long()
    word_bd = collate_1d_or_2d([s for s in word_bd_list], 0).to(ret['ph_frame_logits'].device).long()

    return mel2ph, ph_bd, mel2word, word_bd, ph_of_list, word_of_list, dp_matrix_list
