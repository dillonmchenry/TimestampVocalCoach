import argparse
import os
import json
import shutil
from tqdm import tqdm
from glob import glob
import numpy as np
from utils.os_utils import safe_path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydub import AudioSegment
import torch
from utils.text.zh_text_norm import NSWNormalizer
from utils.text.text_encoder import PUNCS
from utils.text.text_encoder import is_sil_phoneme
from pypinyin import pinyin, Style
import re

class TxtProcessor:
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        if preprocess_args['with_phsep']:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args['add_eos_bos']:
            # remove sil phoneme in head and tail
            while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
                txt_struct = txt_struct[1:]
            while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
                txt_struct = txt_struct[:-1]
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_

    @staticmethod
    def sp_phonemes():
        return ['|', '#']

    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False).lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        text = re.sub(rf"[A-Za-z]+", r"$", text)
        return text

    @classmethod
    def pinyin_with_en(cls, txt, style):
        x = pinyin(txt, style)
        x = [t[0] for t in x]
        x_ = []
        for t in x:
            if '$' not in t:
                x_.append(t)
            else:
                x_ += list(t)
        x_ = [t if t != '$' else 'LANG-ENG' for t in x_]
        x_ = [t if t != '&' else 'BREATHE' for t in x_]
        x_ = [t if t != '@' else '<SEP>' for t in x_]
        return x_

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt)

        shengmu = cls.pinyin_with_en(txt, style=Style.INITIALS)
        yunmu = cls.pinyin_with_en(txt, style=Style.FINALS_TONE3 if preprocess_args['use_tone'] else Style.FINALS)
        assert len(shengmu) == len(yunmu)
        phs = []
        for a, b in zip(shengmu, yunmu):
            if a == b:
                phs += [a]
            else:
                phs += [a + "%" + b]
        if preprocess_args['use_char_as_word']:
            words = list(txt)
        else:
            words = jieba.cut(txt)
        txt_struct = [[w, []] for w in words]
        i_ph = 0
        for ts in txt_struct:
            ts[1] = [ph for char_pinyin in phs[i_ph:i_ph + len(ts[0])]
                     for ph in char_pinyin.split("%") if ph != '']
            i_ph += len(ts[0])
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt

def get_phone(text, txt_process):
    txt_struct, txt = TxtProcessor.process(text, txt_process)
    phs = [p for w in txt_struct for p in w[1]]
    ph_gb_word = ["_".join(w[1]) for w in txt_struct]
    words = [w[0] for w in txt_struct]
    ph2word = [w_id for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
    ph = " ".join(phs) 
    word = " ".join(words)
    ph_gb_word = " ".join(ph_gb_word)
    ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                    for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
    #  Whether to use tones
    if not txt_process['mfa_use_tone']:
        ph_gb_word_nosil = re.sub(r'\d', '', ph_gb_word_nosil)

    return words, phs, ph2word, ph_gb_word_nosil

def remove_emojis_simple(text):
    """Simplified emoji removal (better performance)"""
    # Only remove specific known emojis from the code
    emojis_to_remove = {
        "😊", "😔", "😡", "😰", "🤢", "😮",  # Emoticons
        "🎼", "👏", "😀", "😭", "🤧", "😷",  # Events
        "❓"  # Unknown symbols
    }
    for emoji in emojis_to_remove:
        text = text.replace(emoji, "")
    return text.strip()
def cut_long_audio(wav_path, output_dir, max_duration=20, min_duration=8):
    """
    Split long audio files into segments not exceeding specified duration, 
    while ensuring each segment is at least the minimum duration
    
    :param wav_path: Input audio file path
    :param output_dir: Output directory
    :param max_duration: Maximum segment duration (seconds)
    :param min_duration: Minimum segment duration (seconds)
    :return: List of tuples (output_path, start_time_ms, end_time_ms)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    max_duration_ms = max_duration * 1000
    min_duration_ms = min_duration * 1000
    
    # Return original file if duration is within limit
    if duration_ms <= max_duration_ms:
        base_name = os.path.basename(wav_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_0.wav")
        audio.export(output_path, format="wav")
        return [(output_path, 0, duration_ms)]
    
    # Initialize VAD model
    vad_model = AutoModel(model="fsmn-vad", max_single_segment_time=max_duration_ms, device="cuda:0")
    
    # Detect speech segments using VAD
    try:
        vad_result = vad_model.generate(input=wav_path)
        segments = vad_result[0]["value"]  # Format: [[start1, end1], [start2, end2], ...]
    except Exception as e:
        print(f"VAD processing failed: {str(e)}, using equal segmentation")
        segments = []
    
    # Optimize: Merge consecutive short speech segments
    if segments:
        processed_segments = []
        current_start, current_end = segments[0]  # Initialize first segment
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1][1]
            next_start, next_end = segments[i]
            
            # Calculate current segment duration
            current_duration = current_end - current_start
            
            # Check if merge is needed (merge if either condition is met)
            # 1. Current segment too short (<min_duration)
            # 2. Gap <500ms and merged duration <max_duration
            gap = next_start - prev_end
            merged_duration = next_end - current_start
            
            if (current_duration < min_duration_ms) or \
               (gap < 500 and merged_duration < max_duration_ms):
                current_end = next_end 
            else:
                processed_segments.append([current_start, current_end])
                current_start, current_end = segments[i]
        
        processed_segments.append([current_start, current_end])
        segments = processed_segments
    
    # Use equal segmentation
    if not segments:
        segment_count = int(np.ceil(duration_ms / max_duration_ms))
        segments = []
        for i in range(segment_count):
            start = i * max_duration_ms
            end = min((i + 1) * max_duration_ms, duration_ms)
            segments.append([start, end])
    
    # Split and save audio segments
    results = []
    base_name = os.path.basename(wav_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    
    for i, (start_ms, end_ms) in enumerate(segments):
        # Expand segment reasonably (within boundaries)
        expanded_start = max(0, start_ms - 100)
        expanded_end = min(duration_ms, end_ms + 100)
        
        # Ensure expanded duration doesn't exceed max
        if expanded_end - expanded_start > max_duration_ms:
            # Prioritize keeping end position unchanged
            expanded_start = min(expanded_start, expanded_end - max_duration_ms)
        
        segment = audio[expanded_start:expanded_end]
        output_path = os.path.join(output_dir, f"{base_name_no_ext}_{i}.wav")
        segment.export(output_path, format="wav")
        results.append((output_path, expanded_start, expanded_end))
    
    # Clean up VAD model
    del vad_model
    torch.cuda.empty_cache()
    
    return results

def process(data_dir, tgt_meta_path):
    temp_cut_dir = os.path.join(os.path.dirname(data_dir), "cut_wavs")
    os.makedirs(temp_cut_dir, exist_ok=True)
    
    txt_process = {
        'with_phsep': False,
        'add_eos_bos': False,
        'use_tone': False,
        'use_char_as_word': True,
        'mfa_use_tone': True
    }
    wav_files = glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not wav_files:
        print(f"No WAV files found in directory: {data_dir}")
        return
    
    # Split long audio and collect segments with timing info
    all_cut_segments = []  # Elements: (cut_path, start_ms, end_ms, original_path)
    print(f"Processing {len(wav_files)} audio files, splitting...")
    
    for wav_path in tqdm(wav_files, desc="Splitting audio"):
        try:
            segments = cut_long_audio(wav_path, temp_cut_dir)
            for cut_path, start_ms, end_ms in segments:
                all_cut_segments.append((cut_path, start_ms, end_ms, wav_path))
        except Exception as e:
            print(f"Error splitting file {wav_path}: {str(e)}")
            continue
    
    # 初始化ASR模型
    model_dir = "iic/SenseVoiceSmall"
    model = AutoModel(
        model=model_dir,
        device="cuda:0",
    )
    
    metadata = []
    print(f"Starting ASR recognition on {len(all_cut_segments)} audio segments...")
    
    # Sort by original filename and start time for correct segment order
    all_cut_segments.sort(key=lambda x: (x[3], x[1]))
    
    #  Process each audio segment
    for cut_path, start_ms, end_ms, original_path in tqdm(all_cut_segments, desc="识别音频"):
        try:
            # Perform ASR recognition
            res = model.generate(
                input=cut_path,
                cache={},
                language="auto",
                use_itn=False,
                batch_size_s=60,
                ban_emo_unk=True
            )
            # Post-process recognition results
            if res and res[0].get("text"):
                text = rich_transcription_postprocess(res[0]["text"])
                text = remove_emojis_simple(text)

                word, ph, ph2word, ph_gb_word_nosil = get_phone(text, txt_process)
                original_filename = os.path.basename(original_path)
                
                segment_index = int(os.path.splitext(os.path.basename(cut_path))[0].split('_')[-1])
                
                metadata.append({
                    "item_name": os.path.basename(cut_path)[:-4],
                    "wav_fn": os.path.abspath(cut_path),
                    "text": text,
                    "original_file": original_filename,
                    "segment_index": segment_index,
                    "start_time_ms": start_ms,
                    "end_time_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "word": word,
                    "ph": ph,
                    "ph2words": ph2word,
                    "ph_gb_word_nosil": ph_gb_word_nosil,
                })
        except Exception as e:
            print(f"Error processing file {cut_path}: {str(e)}")
            continue
    
    with open(tgt_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Results saved to: {tgt_meta_path}")
    print(f"Split audio files saved in: {temp_cut_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--audio_dir",
        type=str,
        default="input/wav",
        help='Audio file directory (supports subdirectories)'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="input/metadata.json",
        help='Path for generated metadata file'
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=20,
        help='Maximum duration for single audio segment (seconds)'
    )
    args = parser.parse_args()

    data_dir = args.audio_dir
    tgt_meta_path = safe_path(args.output)
    os.makedirs(os.path.dirname(tgt_meta_path), exist_ok=True)
    
    process(data_dir, tgt_meta_path)