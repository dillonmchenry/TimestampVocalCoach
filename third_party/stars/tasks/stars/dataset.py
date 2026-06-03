import os
import math
import gc
import json

import librosa.feature
import numpy as np
import torch
from tqdm import tqdm
import six

from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d, pad_or_cut_xd
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio import librosa_wav2spec
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from utils.commons.signal import get_filter_1d, get_gaussian_kernel_1d, get_hann_kernel_1d, \
    get_triangle_kernel_1d, add_gaussian_noise
from utils.audio.mel import MelNet
from utils.text.text_encoder import build_token_encoder
def get_soft_label_filter(soft_label_func, win_size, hparams):
    # win_size: ms
    win_size = round(int(win_size) * hparams['audio_sample_rate'] / 1000 / hparams['hop_size'])
    win_size = win_size if win_size % 2 == 1 else win_size + 1  # ensure odd number
    if soft_label_func == 'gaussian':
        sigma = win_size / 3 / 2  # 3sigma range
        kernel = get_gaussian_kernel_1d(win_size, sigma)
        kernel = kernel / kernel.max()  # make sure the middle is 1
    elif soft_label_func == 'hann':
        kernel = get_hann_kernel_1d(win_size, periodic=False)
    elif soft_label_func == 'triangle':
        kernel = get_triangle_kernel_1d(win_size)
    soft_filter = get_filter_1d(kernel, win_size, channels=1)
    return soft_filter

def get_mel_len(wav_len, hop_size):
    return (wav_len + hop_size - 1) // hop_size

class PhoneEncoder(object):
    """Base class for converting from ints to/from human readable strings."""
    def __init__(self, token_list_file=None):
        self.token_list = json.load(open(token_list_file))
        self.token_list = ['<Blank>'] + self.token_list
        self._replace_oov = '<Blank>'
        self._init_vocab(self.token_list) 
    
    def _init_vocab(self, token_list):
        """Initialize vocabulary with tokens from token_generator."""
        self.id_to_token = {}
        self.id_to_token.update(enumerate(token_list))
        # _token_to_id is the reverse of _id_to_token
        self.token_to_id = dict((v, k) for k, v in six.iteritems(self.id_to_token))
    
    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [t if t in self.token_to_id else self._replace_oov for t in tokens]
        ret = [self.token_to_id[tok] for tok in tokens]
        return ret
    
class StarsDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(StarsDataset, self).__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.ph_encoder = PhoneEncoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if items is None and self.avail_idxs is not None:
            ds_names_selected = None
            if prefix in ['train', 'valid'] and self.hparams.get('ds_names_in_training', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_training', '').split(';') + ['']
                print(f'| Iterating training sets to find samples belong to datasets {ds_names_selected[:-1]}')
            elif prefix == 'test' and self.hparams.get('ds_names_in_testing', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_testing', '').split(';') + ['']
                print(f'| Iterating testing sets to find samples belong to datasets {ds_names_selected[:-1]}')
            if ds_names_selected is not None:
                avail_idxs = []
                # somehow, can't use self.indexed_ds beforehand (need to create a temp), otherwise '_pickle.UnpicklingError'
                temp_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
                for idx in tqdm(range(len(self)), total=len(self)):
                    item = temp_ds[self.avail_idxs[idx]]
                    if item.get('ds_name', '') in ds_names_selected:
                        avail_idxs.append(self.avail_idxs[idx])
                print(f'| Chose [{len(avail_idxs)}] samples belonging to the desired datasets from '
                      f'[{len(self.avail_idxs)}] original samples. ({len(avail_idxs) / len(self.avail_idxs) * 100:.2f}%)')
                self.avail_idxs = avail_idxs
        if items is None and prefix == 'train' and self.hparams.get('dataset_downsample_rate', 1.0) < 1.0 \
                and self.avail_idxs is not None:
            ratio = self.hparams.get('dataset_downsample_rate', 1.0)
            orig_len = len(self.avail_idxs)
            tgt_len = round(orig_len * ratio)
            self.avail_idxs = np.random.choice(self.avail_idxs, size=tgt_len, replace=False).tolist()
            print(f'| Downsamping training set with ratio [{ratio * 100:.2f}%], [{tgt_len}] samples of [{orig_len}] samples are selected.')
        if items is None:
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        self.soft_filter = {}
        self.noise_ds = None
        noise_snr = self.hparams.get('noise_snr', '6-20')
        if '-' in noise_snr:
            l, r = noise_snr.split('-')
            self.noise_snr = (float(l), float(r))
        else:
            self.noise_snr = float(noise_snr)
        self.mel_net = MelNet(self.hparams)
        self.lan2label = {'Chinese': 0, 'English': 1, 'Italian': 2, 'French': 3, 'Japanese': 4, 'Spanish': 5, 'German': 6, 'Korean': 7, 'Russian': 8}
        self.gen2label = {'female': 0, 'male': 1}
        self.emo2label={"neutral": 0, "happy": 1, "sad": 2, "angry": 3}
        self.meth2label={"pop": 0, "bel canto": 1}
        self.pace2label={"slow": 0, "moderate": 1, "fast": 2}
        self.range2label={"low": 0, "medium": 1, "high": 2}
        self.techgroup2lbl = {'control':0, 'mixed':1, 'falsetto':2, 'pharyngeal':3, 'glissando':4, 'vibrato':5, 'breathy': 6, 'weak': 7, 'strong':8, 'bubble':9}

        
    def add_noise(self, clean_wav):
        if self.noise_ds is None:   # each instance in multiprocessing must create unique ds object
            self.noise_ds = IndexedDataset(f"{self.hparams['noise_data_dir']}/{self.prefix}")
        noise_idx = np.random.randint(len(self.noise_ds))
        noise_item = self.noise_ds[noise_idx]
        noise_wav = noise_item['feat']

        if type(self.noise_snr) == tuple:
            snr = np.random.rand() * (self.noise_snr[1] - self.noise_snr[0]) + self.noise_snr[0]
        else:
            snr = self.noise_snr
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1)) + 1e-5
        adjusted_noise_rms = clean_rms / (10 ** (snr / 20) + 1e-5)
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
                reduction_rate = max_int16 / mixed.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * reduction_rate
        return mixed

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        wav = item['wav']
        noise_added = np.random.rand() < hparams.get('noise_prob', 0.8)
        if self.prefix == 'test' and not hparams.get('noise_in_test', False):
            noise_added = False
        if noise_added:
            wav = self.add_noise(wav)
        if 'gt_ph' in item:
            ph_length = len(item["gt_ph"])
            ph = " ".join(item["gt_ph"])
            note_ph = " ".join(item["ph"])
            ph2words = torch.LongTensor(item['gt_ph2words'])
        else:
            ph_length = len(item["ph"])
            note_ph = ph = " ".join(item["ph"])
            ph2words = torch.LongTensor(item['ph2words'])
            
        ph = torch.LongTensor(self.ph_encoder.encode(ph))
        mel = self.mel_net(wav).squeeze(0).numpy()
        assert len(mel) == self.sizes[index], (len(mel), self.sizes[index])
        max_frames = hparams['max_frames']
        
        mel2ph_len = sum((item["mel2ph"] > 0).astype(int))
        mel2word_len = sum((item["mel2word"] > 0).astype(int))
        T = min(item['len'], mel2word_len, mel2ph_len, len(item['f0']))
        real_len = T
        T = math.ceil(min(T, max_frames) / hparams['frames_multiple']) * hparams['frames_multiple']
        
        spec = torch.Tensor(mel)[:max_frames]
        spec = pad_or_cut_xd(spec, T, dim=0)
        if 5 < hparams.get('use_mel_bins', hparams['audio_num_mel_bins']) < hparams['audio_num_mel_bins']:
            spec = spec[:, :hparams.get('use_mel_bins', 80)]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "ph": ph,
            "word": item['txt'],
            "ph2words": ph2words,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "note_ph": torch.LongTensor(self.ph_encoder.encode(note_ph))
        }            
        if hparams.get('mel_add_noise', 'none') not in ['none', None] and not noise_added \
                and self.prefix in ['train', 'valid']:
            noise_type, std = hparams.get('mel_add_noise').split(':')
            if noise_type == 'gaussian':
                noisy_mel = add_gaussian_noise(sample['mel'], mean=0.0, std=float(std) * np.random.rand())
                sample['mel'] = torch.clamp(noisy_mel, hparams['mel_vmin'], hparams['mel_vmax'])
        sample["mel2word"] = mel2word = pad_or_cut_xd(torch.LongTensor(item.get("mel2word")), T, 0)
        sample["mel2ph"] = mel2ph = pad_or_cut_xd(torch.LongTensor(item.get("mel2ph")), T, 0)
        special_id = 0
        ph_with_special = torch.cat([torch.tensor([special_id], dtype=torch.long), ph])
        sample["ph_frame"] = ph_frame = torch.gather(ph_with_special, 0, mel2ph)
        sample['mel_nonpadding'] = pad_or_cut_xd(sample['mel_nonpadding'].float(), T, 0)
        
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            # pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            if hparams.get('f0_add_noise', 'none') not in ['none', None] and noise_added \
                    and self.prefix in ['train', 'valid']:
                noise_type, std = hparams.get('f0_add_noise').split(':')
                f0 = torch.FloatTensor(f0)
                if noise_type == 'gaussian':
                    f0[uv == 0] = add_gaussian_noise(f0[uv == 0], mean=0.0, std=float(std) * np.random.rand())
            uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
            f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0, uv))
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        sample["f0"], sample["uv"], sample["pitch_coarse"] = f0, uv, pitch_coarse
        
        sample["word_dur"] = torch.FloatTensor(item['word_durs'][:hparams['max_input_tokens']])
        # make boundary labels for word
        word_bd = torch.zeros_like(mel2word)
        word_bd[1:real_len] = (mel2word[1:real_len] - mel2word[:real_len-1] == 1).float()
        sample["word_bd"] = word_bd.long()
            
        # make boundary labels for phone
        if 'gt_ph_durs' in item:
            sample["ph_dur"] = torch.FloatTensor(item['gt_ph_durs'][:hparams['max_input_tokens']])
        else:
            sample["ph_dur"] = torch.FloatTensor(item['ph_durs'][:hparams['max_input_tokens']])
        ph_bd = torch.zeros_like(mel2ph)
        ph_bd[1:real_len] = (mel2ph[1:real_len] - mel2ph[:real_len-1] == 1).float()
        sample["ph_bd"] = ph_bd.long()

        if hparams.get('use_soft_ph_bd', False) and (hparams.get('soft_ph_bd_func', None) not in ['none', None]):
            if 'ph_bd' not in self.soft_filter:
                soft_label_func, win_size = hparams.get('soft_ph_bd_func', None).split(':')
                self.soft_filter['ph_bd'] = get_soft_label_filter(soft_label_func, int(win_size), hparams)
            ph_bd_soft = ph_bd.clone().detach().float()
            ph_bd_soft[ph_bd.eq(1)] = ph_bd_soft[ph_bd.eq(1)] - 1e-7  # avoid nan
            ph_bd_soft = ph_bd_soft.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                ph_bd_soft = self.soft_filter['ph_bd'](ph_bd_soft).squeeze().detach()
            sample['ph_bd_soft'] = ph_bd_soft
            if hparams.get('ph_bd_add_noise', 'none') not in ['none', None]:
                noise_type, std = hparams.get('ph_bd_add_noise').split(':')
                if noise_type == 'gaussian':
                    noisy_ph_bd_soft = add_gaussian_noise(ph_bd_soft, mean=0.0, std=float(std) * np.random.rand())
                    sample['ph_bd_soft'] = torch.clamp(noisy_ph_bd_soft, 0.0, 1 - 1e-7)
                    
        # delete big redundancy
        if not hparams.get('use_mel', True) and 'mel' in sample:
            del sample['mel']
            
        if not hparams.get('use_wav', False) and 'wav' in sample:
            del sample['wav']
            
        sample['note_num'] = torch.LongTensor(item['note_num'])
        if 'ep_pitches' in item:
            sample["note"] = note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        elif 'pitches' in item:
            sample["note"] = note = torch.LongTensor(item['pitches'][:hparams['max_input_tokens']])

        if hparams.get('use_ph_as_note', True):
            sample["note_dur"] = torch.FloatTensor(item['ph_durs'][:hparams['max_input_tokens']])
            sample["mel2note"] = mel2note = pad_or_cut_xd(torch.LongTensor(item.get("mel2note")), T, 0)
            special_id = 0
            note_with_special = torch.cat([torch.tensor([special_id], dtype=torch.long), note])
            sample["note_frame"] = note_frame = torch.gather(note_with_special, 0, mel2note)
            # make boundary labels for phone
            note_bd = torch.zeros_like(mel2note)
            note_bd[1:real_len] = (mel2note[1:real_len] - mel2note[:real_len-1] == 1).float()
            sample["note_bd"] = note_bd.long()
        else:
            if 'note_durs' in item:
                sample["note_dur"] = torch.FloatTensor(item['note_durs'][:hparams['max_input_tokens']])
            elif 'ep_notedurs' in item:
                sample["note_dur"] = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
            note_bd = torch.zeros_like(mel2word)
            note_dur_ = sample["note_dur"].cumsum(0)[:-1]
            note_bd_idx = torch.round(note_dur_ * hparams['audio_sample_rate'] / hparams["hop_size"]).long()
            note_bd_max_idx = real_len - 1
            note_bd[note_bd_idx[note_bd_idx < note_bd_max_idx]] = 1
            sample["note_bd"] = note_bd.long()
            # deal with truncated note boundaries and the corresponding notes and note durs
            if note_bd.sum() + 1 < len(sample['note']):
                tgt_size = note_bd.sum().item() + 1
                sample['note'] = sample['note'][:tgt_size]
                sample['note_dur'] = sample['note_dur'][:tgt_size]
            sample["mel2note"] = mel2note = torch.cumsum(note_bd, 0) + 1
            note_with_special = torch.cat([torch.tensor([special_id], dtype=torch.long), sample['note']])
            sample["note_frame"] = note_frame = torch.gather(note_with_special, 0, mel2note)

        if hparams.get('use_soft_note_bd', False) and (hparams.get('soft_note_bd_func', None) not in ['none', None]):
            if 'note_bd' not in self.soft_filter:
                soft_label_func, win_size = hparams.get('soft_note_bd_func', None).split(':')
                self.soft_filter['note_bd'] = get_soft_label_filter(soft_label_func, int(win_size), hparams)
            note_bd_soft = note_bd.clone().detach().float()
            note_bd_soft[note_bd.eq(1)] = note_bd_soft[note_bd.eq(1)] - 1e-7  # avoid nan
            note_bd_soft = note_bd_soft.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                note_bd_soft = self.soft_filter['note_bd'](note_bd_soft).squeeze().detach()
            sample['note_bd_soft'] = note_bd_soft
            if hparams.get('note_bd_add_noise', 'none') not in ['none', None]:
                noise_type, std = hparams.get('note_bd_add_noise').split(':')
                if noise_type == 'gaussian':
                    noisy_note_bd_soft = add_gaussian_noise(note_bd_soft, mean=0.0, std=float(std) * np.random.rand())
                    sample['note_bd_soft'] = torch.clamp(noisy_note_bd_soft, 0.0, 1 - 1e-7)
        
        if hparams.get('tech_pre', False):
            item_name = item['item_name']
            tech_group = None
            if len(item_name.split('#')) == 5:
                spk, tech_group, song_name, tech_name, sen_id = item_name.split('#')  # '华为女声#假声#会呼吸的痛#女声_弱假声#0'
            elif len(item_name.split('#')) == 6:
                lan, spk, tech_group, song_name, tech_name, sen_id = item_name.split('#')  #  Chinese#ZH-Alto-1#Breathy#不再见#Breathy_Group#0000
            else:
                tech_name = ''
        
            if 'Control_Group' in tech_name:
                tech_group = 'control'
            elif 'Mixed_Voice_Group' in tech_name:
                tech_group = 'mixed'
            elif 'Falsetto_Group' in tech_name:
                tech_group = 'falsetto'
            elif 'Pharyngeal_Group' in tech_name:
                tech_group = 'pharyngeal'
            elif 'Glissando_Group' in tech_name:
                tech_group = 'glissando'
            elif 'Vibrato_Group' in tech_name:
                tech_group = 'vibrato'
            elif 'Breathy_Group' in tech_name:
                tech_group = 'breathy'
            elif 'Weak_Group' in tech_name:   
                tech_group = 'weak'
            elif 'Strong_Group' in tech_name:   
                tech_group = 'strong'
            elif 'Bubble_Group' in tech_name:   
                tech_group = 'bubble'
            
            if tech_group in self.techgroup2lbl:
                sample['tech_id'] = self.techgroup2lbl[tech_group]
            else:
                sample['tech_id'] = None
            
            mix_tech = torch.LongTensor(item['gt_mix_tech']) if  'gt_mix_tech' in item else torch.ones(ph_length) * -1
            falsetto_tech = torch.LongTensor(item['gt_falsetto_tech']) if  'gt_falsetto_tech' in item else torch.ones(ph_length) * -1
            breathe_tech = torch.LongTensor(item['gt_breathy_tech']) if  'gt_breathy_tech' in item else torch.ones(ph_length) * -1
            bubble_tech = torch.LongTensor(item['gt_bubble_tech']) if  'gt_bubble_tech' in item else torch.ones(ph_length) * -1
            strong_tech = torch.LongTensor(item['gt_strong_tech']) if  'gt_strong_tech' in item else torch.ones(ph_length) * -1
            weak_tech = torch.LongTensor(item['gt_weak_tech']) if  'gt_weak_tech' in item else torch.ones(ph_length) * -1
            pharyngeal_tech = torch.LongTensor(item['gt_pharyngeal_tech']) if  'gt_pharyngeal_tech' in item else torch.ones(ph_length) * -1
            vibrato_tech = torch.LongTensor(item['gt_vibrato_tech']) if  'gt_vibrato_tech' in item else torch.ones(ph_length) * -1
            glissando_tech = torch.LongTensor(item['gt_glissando_tech']) if  'gt_glissando_tech' in item else torch.ones(ph_length) * -1

            tech_matrix = torch.stack((bubble_tech, breathe_tech, pharyngeal_tech, vibrato_tech, glissando_tech, mix_tech, falsetto_tech, weak_tech, strong_tech), dim=1)
            sample['tech'] = tech_matrix    # [T, C]
            
        singer = item['singer']
        if hparams.get('lan_pre', False):
            language = item['language'] if 'emotion' in item else None
            sample['language'] = self.lan2label[language] if language in self.lan2label else None

        if hparams.get('gen_pre', False):
            gender = item['gender'] if 'gender' in item else None
            sample['gender'] = self.gen2label[gender] if gender in self.gen2label else None

        if hparams.get('emo_pre', False):
            emotion = item['emotion'] if 'emotion' in item else None
            sample['emotion'] = self.emo2label[emotion] if emotion in self.emo2label else None

        if hparams.get('meth_pre', False):
            singing_method = item['singing_method'] if 'singing_method' in item else None
            sample['singing_method'] = self.meth2label[singing_method] if singing_method in self.meth2label else None

        if hparams.get('pace_pre', False):
            pace = item['pace'] if 'pace' in item else None
            sample['pace'] = self.pace2label[pace] if pace in self.pace2label else None

        if hparams.get('range_pre', False):
            ranges = item['range'] if 'range' in item else None
            sample['range'] = self.range2label[ranges] if ranges in self.range2label else None
        
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        sample_id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        words = [s['word'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
        ph = collate_1d_or_2d([s['ph'] for s in samples], 0) if 'ph' in samples[0] else None #[s['ph'] for s in samples]
        note_ph = collate_1d_or_2d([s['note_ph'] for s in samples], 0) if 'note_ph' in samples[0] else None
        note_num = collate_1d_or_2d([s['note_num'] for s in samples], 0) if 'note_num' in samples[0] else None
        ph_frame = collate_1d_or_2d([s['ph_frame'] for s in samples], 0) if 'ph' in samples[0] else None
        note_frame = collate_1d_or_2d([s['note_frame'] for s in samples], 0) if 'note_frame' in samples[0] else None
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples]) if 'mel' in samples[0] else 0
        ph_lengths = torch.LongTensor([s['ph'].shape[0] for s in samples]) if 'ph' in samples[0] else 0
        note_ph_lengths = torch.LongTensor([s['note_ph'].shape[0] for s in samples]) if 'note_ph' in samples[0] else 0
        mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None
        ph_nonpadding = collate_1d_or_2d([torch.ones(s['ph'].shape[0]) for s in samples], 0.0) if 'ph' in samples[0] else 0
        note_ph_nonpadding = collate_1d_or_2d([torch.ones(s['note_ph'].shape[0]) for s in samples], 0.0) if 'note_ph' in samples[0] else 0
        
        tech_ids = [s.get('tech_id') for s in samples]
        languages = [s.get('language') for s in samples]
        genders = [s.get('gender') for s in samples]
        emotions = [s.get('emotion') for s in samples]
        singing_methods = [s.get('singing_method') for s in samples]
        paces = [s.get('pace') for s in samples]
        ranges = [s.get('range') for s in samples]

        batch = {
            'id': sample_id,
            'item_name': item_names,
            'nsamples': len(samples),
            "ph": ph,
            'words': words,
            "ph_lengths": ph_lengths,
            "note_ph": note_ph,
            "note_ph_lengths": note_ph_lengths,
            'note_ph_nonpadding': note_ph_nonpadding,
            'ph_frame': ph_frame,
            'note_frame': note_frame,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel_nonpadding': mel_nonpadding,
            'ph_nonpadding': ph_nonpadding,
            'tech_ids': tech_ids,
            'languages': languages,
            'genders': genders,
            'emotions': emotions,
            'singing_methods': singing_methods,
            'paces': paces,
            'ranges': ranges,
            'note_num': note_num
        }

        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch_coarse = collate_1d_or_2d([s['pitch_coarse'] for s in samples])
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        batch['f0'], batch['uv'], batch['pitch_coarse'] = f0, uv, pitch_coarse
        batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None

        batch['mel2ph'] = collate_1d_or_2d([s['mel2ph'] for s in samples], 0)
        batch['mel2note'] = collate_1d_or_2d([s['mel2note'] for s in samples], 0)
        batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        
        batch['ph2words'] = collate_1d_or_2d([s['ph2words'] for s in samples], 0)
        batch["word_durs"] = collate_1d_or_2d([s['word_dur'] for s in samples], 0.0)
        batch["ph_durs"] = collate_1d_or_2d([s['ph_dur'] for s in samples], 0.0)
        batch["word_bd"] = collate_1d_or_2d([s['word_bd'] for s in samples], 0.0)
        batch["ph_bd"] = collate_1d_or_2d([s['ph_bd'] for s in samples], 0.0)
        batch["ph_bd_soft"] = collate_1d_or_2d([s['ph_bd_soft'] for s in samples], 0.0) if 'ph_bd_soft' in samples[0] else None
        batch['techs'] = collate_1d_or_2d([s['tech'] for s in samples], 0.0) if 'tech' in samples[0] else None
        batch["notes"] = collate_1d_or_2d([s['note'] for s in samples], 0.0) if 'note' in samples[0] else None
        batch["note_nonpadding"] = collate_1d_or_2d([torch.ones(s['note'].shape[0]) for s in samples], 0.0) if 'note' in samples[0] else 0
        batch["word_nonpadding"] = collate_1d_or_2d([torch.ones(s['mel2word'].max()) for s in samples], 0.0) if 'mel2word' in samples[0] else 0
        batch["note_durs"] = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        batch["note_bd"] = collate_1d_or_2d([s['note_bd'] for s in samples], 0.0) if 'note_bd' in samples[0] else None
        batch["note_bd_soft"] = collate_1d_or_2d([s['note_bd_soft'] for s in samples], 0.0) if 'note_bd_soft' in samples[0] else None

        return batch

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))
    
if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.dataset_utils import data_loader, BaseConcatDataset, build_dataloader

    hparams = set_hparams(config='/configs/zh/baseline.yaml')
    train_dataset = StarsDataset(prefix=hparams['train_set_name'], shuffle=True)
    data_loader = build_dataloader(train_dataset, True, hparams['max_tokens'], 1)
    for batch in data_loader:
        # break
        pass