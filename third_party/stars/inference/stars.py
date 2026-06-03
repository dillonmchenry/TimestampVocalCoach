import os
import argparse
from pathlib import Path
import json
import math
from collections import defaultdict
import traceback
import sys
import textgrid

import librosa
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group
import torch.distributed as dist
import matplotlib.pyplot as plt
from utils.audio.align import mel2token_to_dur

from utils.os_utils import safe_path
from utils.commons.hparams import set_hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.dataset_utils import batch_by_size, pad_or_cut_xd, collate_1d_or_2d, build_dataloader
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import move_to_cuda
from utils.audio import get_wav_num_frames
from utils.audio.mel import MelNet
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval, save_midi, midi_to_hz

from modules.pe.rmvpe import RMVPE
from tasks.stars.dataset import get_mel_len, PhoneEncoder
from tasks.stars.utils import bd_to_durs, regulate_real_note_itv, regulate_ill_slur
from modules.stars.stars import STARS
from data_gen.base_binarizer import BaseBinarizer, BinarizationError
from utils.audio.pitch_extractors import get_pitch_extractor

def align_word(word_durs, mel_len, hop_size, audio_sample_rate):
    mel2word = np.zeros([mel_len], int)
    start_time = 0
    for i_word in range(len(word_durs)):
        start_frame = int(start_time * audio_sample_rate / hop_size + 0.5)
        end_frame = int((start_time + word_durs[i_word]) * audio_sample_rate / hop_size + 0.5)
        if start_frame == end_frame:
            raise BinarizationError(f"Zero duration for word {i_word + 1} (start_frame: {start_frame}, end_frame: {end_frame})")

        mel2word[start_frame:end_frame] = i_word + 1
        start_time = start_time + word_durs[i_word]

    dur_word = mel2token_to_dur(mel2word)

    return mel2word, dur_word.tolist()

def align_ph(ph_durs, mel_len, hop_size, audio_sample_rate):
    mel2ph = np.zeros([mel_len], int)
    start_time = 0
    for i_ph in range(len(ph_durs)):
        start_frame = int(start_time * audio_sample_rate / hop_size + 0.5)
        end_frame = int((start_time + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
        if start_frame == end_frame:
            raise BinarizationError(f"Zero duration for phone {i_ph + 1} (start_frame: {start_frame}, end_frame: {end_frame})")

        mel2ph[start_frame:end_frame] = i_ph + 1
        start_time = start_time + ph_durs[i_ph]

    dur_ph = mel2token_to_dur(mel2ph)

    return mel2ph, dur_ph.tolist()

def get_textgrid(mel_len, ph_of, word_of, tech_pred, hop_size_second=0.02, draw=False, id2token=None, word_list=None, dp_matrix=None, mel=None, tg_fn=None):
    minTime = 0
    maxTime = max(mel_len.cpu().numpy(), word_of[-1][1], ph_of[-1][1]) * hop_size_second
    tg = textgrid.TextGrid(minTime=minTime, maxTime=maxTime)
    word_intervals = textgrid.IntervalTier(name="words", minTime=minTime, maxTime=maxTime)
    phone_intervals = textgrid.IntervalTier(name="phones", minTime=minTime, maxTime=maxTime)
    tech_intervals = {}
    tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mixed', 'falsetto', 'weak', 'strong']
    for i, name in enumerate(tech_names):
        tech_intervals[name] = textgrid.IntervalTier(name=name, minTime=minTime, maxTime=maxTime)

    for word in word_of:
        word_id = word[2]
        minTime = word[0] * hop_size_second
        maxTime = word[1] * hop_size_second
        if word_id == -1:
            mark = '<SP>'
        else:
            mark = word_list[word_id]
        tg_word = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=mark)
        word_intervals.addInterval(tg_word)

    for idx, phone in enumerate(ph_of):
        ph_id = phone[2]
        minTime = phone[0] * hop_size_second
        maxTime = phone[1] * hop_size_second
        mark = id2token[ph_id]
        tg_phone = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=mark)
        phone_intervals.addInterval(tg_phone)
        for i, name in enumerate(tech_names):
            tg_tech = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=str(tech_pred[idx][i].item()))
            tech_intervals[name].addInterval(tg_tech)

    tg.tiers.append(word_intervals)
    tg.tiers.append(phone_intervals)
    for i, name in enumerate(tech_names):
        tg.tiers.append(tech_intervals[name])

    tg.write(tg_fn)

    if draw: 
        plt.style.use('default')
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[0.2, 2, 1], hspace=0.08)

        ax_title = fig.add_subplot(gs[0])
        ax_mel = fig.add_subplot(gs[1])
        ax_dp = fig.add_subplot(gs[2])

        ax_title.axis('off')
        
        if mel is not None:
            mel = mel.numpy() if isinstance(mel, torch.Tensor) else mel
            time_steps, n_mels = mel.shape
            
            im_mel = ax_mel.imshow(mel.T, origin='lower', aspect='auto',
                                cmap='viridis', interpolation='nearest')
            
            cbar = plt.colorbar(im_mel, ax=ax_mel, fraction=0.02, pad=0.01)
            cbar.set_label('Energy (dB)', rotation=270, labelpad=15)

            title_text = f"Phoneme Alignment Visualization"
            ax_title.text(0.5, 0.5, title_text, 
                        ha='center', va='center', 
                        fontsize=14, color='navy')

            label_params = {
                'base_y': n_mels * 1.00,
                'line_height': - n_mels * 0.04,
                'max_lines': 2,  
                'char_width': 0.1,  
                'min_gap': 0.15,
                'fontsize': 10,
            }

            def format_label(ph_id):
                text = id2token[ph_id]
                return text.replace('<', '').replace('>', '')

            sorted_ph = sorted(ph_of, key=lambda x: x[0])
            timeline = [[] for _ in range(label_params['max_lines'])]
            
            for idx, ph in enumerate(sorted_ph):
                start, end, ph_id, _ = ph
                text = format_label(ph_id)
                mid = (start + end) / 2
                
                if all(abs(mid - prev_mid) > label_params['min_gap'] for prev_mid in timeline[0]):
                    current_line = 0
                    timeline[current_line].append(mid)
                else:
                    current_line = 1
                    timeline[current_line].append(mid)

                y_pos = label_params['base_y'] + current_line * label_params['line_height']
                
                ax_mel.text(mid, y_pos, text,
                            color='darkred',
                            ha='center', va='bottom',
                            fontsize=label_params['fontsize'], weight='bold',
                            bbox=dict(facecolor='white', alpha=0.9,
                                    edgecolor='none', lw=0.3,
                                    boxstyle='round,pad=0.1'))

                ax_mel.axvline(start, color='white', ls='-', lw=0.8, alpha=0.8)
                ax_mel.axvline(end, color='white', ls='-', lw=0.8, alpha=0.8)

        dp_matrix_np = np.array(dp_matrix)
        
        valid_mask = dp_matrix_np > -1e7
        dp_valid = np.where(valid_mask, dp_matrix_np, np.nan)
        
        vmin = np.nanpercentile(dp_valid, 5)
        vmax = np.nanpercentile(dp_valid, 95)
        
        im_dp = ax_dp.imshow(dp_valid.T, aspect='auto', origin='lower',
                            cmap='plasma', vmin=vmin, vmax=vmax,
                            interpolation='nearest')
        
        cbar_dp = plt.colorbar(im_dp, ax=ax_dp, fraction=0.02, pad=0.01)
        cbar_dp.set_label('Log Probability', rotation=270, labelpad=15)
        
        prev_state_pos = 0
        num_phone = 0
        for ph_idx, ph in enumerate(ph_of):
            start_frame, end_frame, ph_id, _ = ph
            if ph_id != 1:
                state_pos = num_phone * 2 + 1
                num_phone += 1
            else:
                state_pos = num_phone * 2
            
            ax_dp.hlines(state_pos, start_frame, end_frame,
                        colors='darkred', linewidths=2, 
                        alpha=0.9, linestyle='-')
            
            ax_dp.vlines(start_frame, prev_state_pos-0.25, state_pos+0.25, 
                        colors='darkred', linewidths=2, 
                        alpha=0.9, linestyle='-')
            
            prev_state_pos = state_pos
        time_formatter = lambda x, _: f"{x*hop_size_second:.2f}s"
        
        ax_mel.set_xlim(0, dp_matrix_np.shape[0])
        ax_mel.xaxis.set_major_formatter(time_formatter)
        ax_mel.set_ylabel("Mel Bins")

        ax_dp.set_xlabel("Time (seconds)")
        ax_dp.set_ylabel("Viterbi States")
        ax_dp.xaxis.set_major_formatter(time_formatter)

        output_img_path = tg_fn.replace('.TextGrid', '_alignment.png')
        plt.savefig(output_img_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {output_img_path}")
        plt.close()
    return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        default='infer_out',
        help='Directory of outputs. '
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='checkpoints/stars_chinese/model_ckpt_steps_200000.ckpt',
        help='Path of asr ckpt. '
    )
    parser.add_argument(
        "--config",
        type=str,
        default='configs/stars_chinese.yaml',
        help='Path of config file. If not provided, will be inferred under the same directory of ckpt. '
    )
    parser.add_argument(
        "--phset",
        type=str,
        default='chinese_phone_set.json',
        help='Path of phone_set file. '
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default='input/metadata.json',
        help='Path of the metadata of the desired input data. '
             'The metadata should be a .json file containing a list of dicts, where each dicts should contain'
             'attributes: "item_name", "wav_fn", "ph", and "ph2word". '
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.85,
        help='Threshold to determine note boundaries. '
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="Whether or not to print detailed information. "
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=128,
        help='Batch size (max sentences) for each node. '
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100000,
        help='Max tokens for each node. '
    )
    parser.add_argument(
        "--ds_workers",
        type=int,
        default=1,
        help='Number of workers to generate samples. Set to 0 for single inference. '
    )
    parser.add_argument(
        "--save_plot",
        action='store_true',
        help='Save the plots of MIDI or not. '
    )
    parser.add_argument(
        "--no_save_textgrid",
        action='store_true',
        help='Save TextGrid files or not. '
    )
    parser.add_argument(
        "--check_slur",
        action='store_true',
        help='Check appearances of slurs and print logs.'
    )
    parser.add_argument(
        "--no_save_midi",
        action='store_true',
        help="Don't save MIDI files. "
    )
    parser.add_argument(
        "--sync_saving",
        action='store_true',
        help="Synchronized results saving. "
    )
    args = parser.parse_args()
        
    return args

class AlignInfer:
    def __init__(self, num_gpus=-1):
        self.args = parse_args()
        self.work_dir = self.args.save_dir
        if num_gpus == -1:
            all_gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
            self.num_gpus = len(all_gpu_ids)
        else:
            self.num_gpus = num_gpus
        self.hparams = {}
        self.tech_threshold = self.args.thr
        
        # style dict
        self.lan2label = {'Chinese': 0, 'English': 1, 'Italian': 2, 'French': 3, 
                         'Japanese': 4, 'Spanish': 5, 'German': 6, 'Korean': 7, 'Russian': 8}
        self.gen2label = {'female': 0, 'male': 1}
        self.emo2label = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3}
        self.meth2label = {"pop": 0, "bel canto": 1}
        self.pace2label = {"slow": 0, "moderate": 1, "fast": 2}
        self.range2label = {"low": 0, "medium": 1, "high": 2}
        self.techgroup2lbl = {'control':0, 'mixed':1, 'falsetto':2, 'pharyngeal':3, 
                              'glissando':4, 'vibrato':5, 'breathy': 6, 'weak': 7, 
                              'strong':8, 'bubble':9}
        
        # reverse
        self.label2lan = {v: k for k, v in self.lan2label.items()}
        self.label2gen = {v: k for k, v in self.gen2label.items()}
        self.label2emo = {v: k for k, v in self.emo2label.items()}
        self.label2meth = {v: k for k, v in self.meth2label.items()}
        self.label2pace = {v: k for k, v in self.pace2label.items()}
        self.label2range = {v: k for k, v in self.range2label.items()}
        self.lbl2techgroup = {v: k for k, v in self.techgroup2lbl.items()}

    def build_model(self, device=None, verbose=True):
        model = STARS(self.hparams)
        self.global_steps = load_ckpt(model, self.args.ckpt, verbose=verbose, return_steps=True)
        model.eval()
        self.ph_encoder = PhoneEncoder(self.args.phset)
        if device is not None:
            model.to(device)
        return model

    def run(self):
        ckpt_path = self.args.ckpt
        config_path = Path(ckpt_path).with_name('config.yaml') if self.args.config == '' else self.args.config
        self.hparams = set_hparams(
            config=config_path,
            print_hparams=self.args.verbose,
            hparams_str=f"note_bd_threshold={self.args.thr}"
        )

        items = json.load(open(self.args.metadata))
        results = []
        if self.num_gpus > 1:
            result_queue = mp.Queue()
            for rank in range(self.num_gpus):
                mp.Process(target=self.run_worker, args=(rank, items, self.args.ds_workers, not self.args.sync_saving, result_queue,)).start()
            for _ in range(self.num_gpus):
                results_ = result_queue.get()
                results.extend(results_)
        else:
            results = self.run_worker(0, items, self.args.ds_workers, False, None)

        self.after_infer(results)

    @torch.no_grad()
    def run_worker(self, rank, items, ds_workers=0, async_save_result=True, q=None):
        if self.num_gpus > 1:
            init_process_group(backend="nccl", init_method="tcp://localhost:54189",
                               world_size=self.num_gpus, rank=rank)

        # build models
        device = torch.device(f"cuda:{int(rank)}")
        # build f0 model
        if self.hparams['use_pitch_embed']:
            pe = RMVPE(self.hparams['pe_ckpt'], device=device)
        # build main model
        model= self.build_model(device, verbose=rank == 0)

        # build dataset
        dataset = AlignInferDataset(items, self.hparams, ds_workers)
        loader = build_dataloader(dataset, shuffle=False, max_tokens=self.args.max_tokens, max_sentences=self.args.bsz, use_ddp=self.num_gpus > 1)
        loader = tqdm(loader, desc=f"| Generating in [n_ranks={self.num_gpus}; "
                                   f"max_tokens={self.args.max_tokens}; "
                                   f"max_sentences={self.args.bsz}]") if rank == 0 else loader
        os.makedirs(os.path.join(self.work_dir, 'textgrid'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'midi'), exist_ok=True)
        # results queue
        saving_result_pool = MultiprocessManager(int(ds_workers))
        results = []

        # run main inference
        with torch.no_grad():
            for batch in loader:
                if batch is None or len(batch) == 0:
                    continue
                batch = move_to_cuda(batch, int(rank))
                bsz = batch['nsamples']

                # get f0
                if self.hparams['use_pitch_embed']:
                    if self.hparams.get('use_rmvpe', False):
                        f0s, uvs = pe.get_pitch_batch(
                            batch['wav'], sample_rate=self.hparams['audio_sample_rate'], hop_size=self.hparams['hop_size'],
                            lengths=batch['real_lens'], fmax=self.hparams['f0_max'], fmin=self.hparams['f0_min']
                        )
                    else:
                        f0s, uvs = [], []
                        for wav in batch['wav']:
                            wav_np = wav.cpu().numpy()
                            f0 = get_pitch_extractor('parselmouth')(wav_np, audio_sample_rate=self.hparams['audio_sample_rate'], hop_size=self.hparams['hop_size'], f0_max=self.hparams['f0_max'], f0_min=self.hparams['f0_min'])
                            uv = (f0 > 0).astype(np.float32)
                            f0s.append(f0)
                            uvs.append(uv)
                        
                    f0_batch, uv_batch, pitch_batch = [], [], []
                    for i, (f0, uv) in enumerate(zip(f0s, uvs)):
                        T = batch['lens'][i]
                        f0, uv = norm_interp_f0(f0[:T])
                        f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
                        f0_batch.append(f0)
                        uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
                        uv_batch.append(uv)
                        pitch_batch.append(f0_to_coarse(denorm_f0(f0, uv)))
                    batch["f0"] = f0 = collate_1d_or_2d(f0_batch).to(device)
                    batch["uv"] = uv = collate_1d_or_2d(uv_batch).long().to(device)
                    batch["pitch_coarse"] = pitch_coarse = collate_1d_or_2d(pitch_batch).to(device)
                else:
                    batch["f0"] = f0 = batch["uv"] = uv = batch["pitch_coarse"] = pitch_coarse = None

                mel_input = batch['mels']
                word_bd = batch['word_bd']
                ph_bd = batch['ph_bd']
                ph = batch['ph']
                ph_lengths = batch['ph_lengths']
                mel_lengths = batch['mel_lengths']
                pitch_coarse = batch['pitch_coarse']
                uv = batch['uv'].long()
                mel2ph = batch['mel2ph']
                mel2word = batch['mel2word']
                f0_denorm = denorm_f0(batch['f0'], batch["uv"])
                bsz = batch['nsamples']
                mel_nonpadding = batch['mel_nonpadding']
                ph2words = batch['ph2words']
                output = model(mel=mel_input, pitch=pitch_coarse, uv=uv, mel_nonpadding=mel_nonpadding, mel2ph=mel2ph, mel2word=mel2word, ph_bd=ph_bd, word_bd=word_bd, ph=ph, ph_lengths=ph_lengths, mel_lengths=mel_lengths, ph2words=ph2words, train=False, global_steps=self.global_steps)
                                
                # global style
                technique_pred = torch.argmax(output['technique_logits'], dim=-1)
                language_pred = torch.argmax(output['language_logits'], dim=-1)
                gender_pred = torch.argmax(output['gender_logits'], dim=-1)
                emotion_pred = torch.argmax(output['emotion_logits'], dim=-1)
                method_pred = torch.argmax(output['method_logits'], dim=-1)
                pace_pred = torch.argmax(output['pace_logits'], dim=-1)
                range_pred = torch.argmax(output['range_logits'], dim=-1)
                
                for idx in range(bsz):
                    item_name = batch['item_name'][idx]
                    mel = batch['mels'][idx].data.cpu()[:mel_lengths[idx]]
                    ph_of = output['ph_of_list'][idx]
                    word_of = output['word_of_list'][idx]
                    dp_matrix = output['dp_matrix_list'][idx]
                    word_list = batch['words'][idx]
                    ph_list = batch['ph_list'][idx]
                    
                    tech_logits = output['tech_logits'][idx]
                    tech_pred = torch.sigmoid(tech_logits)  # [B, ph_length, tech_num]
                    tech_pred = (tech_pred > self.tech_threshold).long().cpu()
                    ph_len = len(ph_of)
                    
                    # save TextGrid (optional)
                    if not self.args.no_save_textgrid and ph_bd==None:
                        get_textgrid(
                            mel_lengths[idx], ph_of, word_of, tech_pred[:ph_len], 
                            draw=self.args.save_plot, 
                            hop_size_second=self.hparams['hop_size'] / self.hparams['audio_sample_rate'], 
                            id2token=self.ph_encoder.id_to_token, 
                            word_list=word_list, 
                            mel=mel, 
                            dp_matrix=dp_matrix, 
                            tg_fn=os.path.join(self.work_dir, 'textgrid', f'{item_name}.TextGrid')
                        )
                        word_list = ['<SP>' if word[2] == -1 else word_list[word[2]] for word in word_of]
                        ph_list = [self.ph_encoder.id_to_token[phone[2]] for phone in ph_of]

                    # process note boundary
                    note_bd_pred = output['note_bd_pred'][idx][:mel_lengths[idx]]
                    note_length = output['note_lengths'][idx]
                    note_pred = output['note_pred'][idx][:note_length]
                    word_bd = output['word_bd'][idx][:mel_lengths[idx]]

                    # save MIDI (optional)）
                    if not self.args.no_save_midi:
                        base_fn = f'{item_name}[%s]'.replace(' ', '_')
                        self.save_note(
                            item_name, base_fn, note_bd_pred, note_pred, word_bd=word_bd, 
                            draw=self.args.save_plot, 
                            gt_f0=f0_denorm[idx][:mel_lengths[idx]].cpu()
                        )
                    
                    # result
                    result = {
                        'item_name': item_name,
                        'wav_fn': batch['wav_fn'][idx],
                        'ph_list': ph_list,
                        'word_list': word_list,
                        'note_list': note_pred.tolist(),
                        'bubble_tech': [],
                        'breathe_tech': [],
                        'pharyngeal_tech': [],
                        'vibrato_tech': [],
                        'glissando_tech': [],
                        'mix_tech': [],
                        'falsetto_tech': [],
                        'weak_tech': [],
                        'strong_tech': [],
                        'style': {
                            'language': self.label2lan.get(language_pred[idx].item(), 'unknown'),
                            'gender': self.label2gen.get(gender_pred[idx].item(), 'unknown'),
                            'emotion': self.label2emo.get(emotion_pred[idx].item(), 'unknown'),
                            'method': self.label2meth.get(method_pred[idx].item(), 'unknown'),
                            'pace': self.label2pace.get(pace_pred[idx].item(), 'unknown'),
                            'range': self.label2range.get(range_pred[idx].item(), 'unknown'),
                            'technique_group': self.lbl2techgroup.get(technique_pred[idx].item(), 'unknown')
                        }
                    }
                    
                    # get ph_durs and word_durs
                    hop_size_second = self.hparams['hop_size'] / self.hparams['audio_sample_rate']
                    ph_durs = [(ph[1] - ph[0]) * hop_size_second for ph in ph_of]
                    word_durs = [(word[1] - word[0]) * hop_size_second for word in word_of]
                    
                    result['ph_durs'] = ph_durs
                    result['word_durs'] = word_durs
                    
                    # get note_durs
                    note_itv_pred = boundary2Interval(note_bd_pred.cpu().numpy())
                    note_durs = [(end - start) * hop_size_second for start, end in note_itv_pred]
                    result['note_durs'] = note_durs
                    
                    # pred technique
                    tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mixed', 'falsetto', 'weak', 'strong']
                    for i, name in enumerate(tech_names):
                        tech_key = f"{name}_tech"
                        result[tech_key] = tech_pred[:len(ph_of), i].cpu().numpy().astype(int).tolist()
                    
                    results.append(result)
                    
        # save_result
        output_json_path = os.path.join(self.work_dir, 'output.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            if self.args.verbose:
                print(f"Saved results to {output_json_path}")
                
        if q is not None:
            q.put(results)
        else:
            return results

    def save_note(self, item_name, base_fn, note_bd_pred, note_pred, word_bd, draw=False, gt_f0=None):
        fn = base_fn % 'P'
        note_bd_pred = note_bd_pred.data.cpu().numpy()
        note_pred = note_pred.data.cpu().numpy()
        word_bd = word_bd.data.cpu().numpy()
        note_itv_pred = boundary2Interval(note_bd_pred)
        pred_fn = f'{self.work_dir}/midi/{fn}.mid'
        if self.hparams.get('infer_regulate_real_note_itv', True):
            try:
                word_durs = np.array(bd_to_durs(word_bd)) * self.hparams['hop_size'] / self.hparams['audio_sample_rate']
                note_itv_pred_secs, note2words = regulate_real_note_itv(note_itv_pred, note_bd_pred, word_bd, word_durs, self.hparams['hop_size'], self.hparams['audio_sample_rate'])
                regulate_note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(note_pred, note_itv_pred_secs, note2words)
                save_midi(regulate_note_pred, note_itv_pred_secs, safe_path(pred_fn))
                if self.args.check_slur:
                    check_slur_cnt(note2words, item_name, verbose=self.args.verbose)
            except AssertionError as e:
                print(e)
                note_itv_pred_secs = note_itv_pred * self.hparams['hop_size'] / self.hparams['audio_sample_rate']
                save_midi(note_pred, note_itv_pred_secs, safe_path(pred_fn))
                note2words = None
        else:
            note_itv_pred_secs = note_itv_pred * self.hparams['hop_size'] / self.hparams['audio_sample_rate']
            save_midi(note_pred, note_itv_pred_secs, safe_path(pred_fn))
            note2words = None
        if draw:
            fig = plt.figure()
            plt.plot(gt_f0, color='blue', label='gt f0', lw=1)
            midi_pred = np.zeros(note_bd_pred.shape[0])
            for i, itv in enumerate(np.round(note_itv_pred).astype(int)):
                midi_pred[itv[0]: itv[1]] = note_pred[i]
            midi_pred = midi_to_hz(midi_pred)
            plt.plot(midi_pred, color='green', label='pred midi')
            plt.legend()
            plt.tight_layout()
            os.makedirs(f'{self.work_dir}/midi', exist_ok=True)
            plt.savefig(f'{self.work_dir}/midi/{fn}.png', format='png')
            plt.close(fig)
            if self.args.verbose:
                print(f"Visualization saved to {self.work_dir}/midi/{fn}.png")
        return None

    def after_infer(self, results):
        success_count = len(results)
        total_count = len(json.load(open(self.args.metadata)))
        
        print("\n" + "="*50)
        print(f"Alignment completed successfully!")
        print(f"Total items processed: {total_count}")
        print(f"Successfully aligned items: {success_count}")
        print(f"Success rate: {success_count/total_count*100:.2f}%")
        print("="*50)
        
        summary = {
            "total_items": total_count,
            "success_items": success_count,
            "success_rate": success_count/total_count,
            "details": [{
                "item_name": res['item_name'],
                "status": "success"
            } for res in results]
        }
        
        summary_path = os.path.join(self.work_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Saved summary to {summary_path}")

class AlignInferDataset(Dataset):
    def __init__(self, items, hparams, num_workers):
        self.sizes = []
        self.hparams = hparams
        self.hop_size = hparams['hop_size']
        self.sr = hparams['audio_sample_rate']
       
        for idx, item in enumerate(items):
            total_frames = get_wav_num_frames(item['wav_fn'], self.sr)
            self.sizes.append(get_mel_len(total_frames, self.hop_size))
        
        self.num_workers = num_workers
        self.items = items
        self.ph_encoder = PhoneEncoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
        self.mel_net = MelNet(self.hparams)

    def __getitem__(self, idx):
        hparams = self.hparams
        item = self.items[idx]        
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, _ = librosa.core.load(wav_fn, sr=self.sr)
        mel_len = get_mel_len(wav.shape[-1], self.hop_size)

        ph_length = len(item["ph"])
        ph = " ".join(item["ph"])
        ph = torch.LongTensor(self.ph_encoder.encode(ph))
        ph2words = torch.LongTensor(item['ph2words'])
        mel = self.mel_net(wav).squeeze(0).numpy()

        sample = {
            "id": idx,
            "item_name": item_name,
            "wav_fn": wav_fn,
            "ph": ph,
            "ph_list": item["ph"],
            "word": item['word'],
            "words": item['word'],
            "ph2words": ph2words,
            'wav': torch.from_numpy(wav)
        }

        if 'ph_durs' in item and 'word_durs' in item:
            ph_durs = item['ph_durs']
            word_durs = item['word_durs']
            sample["ph_dur"] = torch.FloatTensor(ph_durs)
            sample["word_dur"] = torch.FloatTensor(word_durs)

            mel2ph, _ = align_ph(ph_durs, mel.shape[0], hparams['hop_size'], hparams['audio_sample_rate'])
            if mel2ph[0] == 0:    # better start from 1, consistent with mel2ph
                mel2ph = [i + 1 for i in mel2ph]

            mel2word, _ = align_word(word_durs, mel.shape[0], hparams['hop_size'], hparams['audio_sample_rate'])
            if mel2word[0] == 0:    # better start from 1, consistent with mel2ph
                mel2word = [i + 1 for i in mel2word]
            
            mel2ph_len = sum((mel2ph > 0).astype(int))
            mel2word_len = sum((mel2word > 0).astype(int))
            real_len = T = min(mel2word_len, mel2ph_len)
            T = math.ceil(T / hparams['frames_multiple']) * hparams['frames_multiple']
            sample["mel2word"] = mel2word = pad_or_cut_xd(torch.LongTensor(mel2word), T, 0)
            sample["mel2ph"] = mel2ph = pad_or_cut_xd(torch.LongTensor(mel2ph), T, 0)

            word_bd = torch.zeros_like(mel2word)
            word_bd[1:real_len] = (mel2word[1:real_len] - mel2word[:real_len-1] == 1).float()
            sample["word_bd"] = word_bd.long()
                
            ph_bd = torch.zeros_like(mel2ph)
            ph_bd[1:real_len] = (mel2ph[1:real_len] - mel2ph[:real_len-1] == 1).float()
            sample["ph_bd"] = ph_bd.long()
        else:
            real_len = T = mel.shape[0]
            T = math.ceil(T / hparams['frames_multiple']) * hparams['frames_multiple']

        sample['real_len'] = real_len
        sample['len'] = T

        spec = pad_or_cut_xd(torch.Tensor(mel), T, dim=0)
        sample['mel'] = spec = spec[:, :hparams.get('use_mel_bins', 80)]
        sample['mel_nonpadding'] = pad_or_cut_xd((spec.abs().sum(-1) > 0).float(), T, 0)
        
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        sample_id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        wav_fns = [s['wav_fn'] for s in samples]
        words = [s['word'] for s in samples]
        ph_lists = [s['ph_list'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
        ph = collate_1d_or_2d([s['ph'] for s in samples], 0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        ph_lengths = torch.LongTensor([s['ph'].shape[0] for s in samples])
        mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0)
        ph2words = collate_1d_or_2d([s['ph2words'] for s in samples], 0)
        batch = {
            'id': sample_id,
            'item_name': item_names,
            'wav_fn': wav_fns,
            'nsamples': len(samples),
            "ph": ph,
            'ph_list': ph_lists,
            'words': words,
            "ph_lengths": ph_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel_nonpadding': mel_nonpadding,
            'real_lens': [s['real_len'] for s in samples],
            'lens': [s['len'] for s in samples],
            'ph2words': ph2words
        }

        batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None
        batch['mel2ph'] = collate_1d_or_2d([s['mel2ph'] for s in samples], 0) if 'mel2ph' in samples[0] else None
        batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0) if 'mel2word' in samples[0] else None
        
        batch["word_durs"] = collate_1d_or_2d([s['word_dur'] for s in samples], 0.0) if 'word_dur' in samples[0] else None
        batch["ph_durs"] = collate_1d_or_2d([s['ph_dur'] for s in samples], 0.0) if 'ph_dur' in samples[0] else None
        batch["word_bd"] = collate_1d_or_2d([s['word_bd'] for s in samples], 0.0) if 'word_bd' in samples[0] else None
        batch["ph_bd"] = collate_1d_or_2d([s['ph_bd'] for s in samples], 0.0) if 'ph_bd' in samples[0] else None

        return batch

    def __len__(self):
        return len(self.items)

    def ordered_indices(self):
        return (np.arange(len(self))).tolist()

    def num_tokens(self, idx):
        return self.sizes[idx]

def check_slur_cnt(note2words, item_name=None, verbose=False):
    cnt = 1
    slur_cnt = defaultdict(int)
    for note_idx in range(1, len(note2words)):
        if note2words[note_idx] == note2words[note_idx - 1]:
            cnt += 1
        else:
            if cnt > 1:
                if cnt > 2 and verbose:
                    print(f"warning: item [{item_name}] has {cnt} notes to 1 word.")
                slur_cnt[cnt] += 1
            cnt = 1
    return slur_cnt

if __name__ == '__main__':
    mp.set_start_method('spawn')
    alignment = AlignInfer()
    alignment.run()