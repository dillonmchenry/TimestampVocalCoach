# -*- coding: utf-8 -*-
"""
text2phoneme_mixed_fixed.py

A single-file script that converts mixed Chinese + Latin (English/German/...) sentences
into a unified token→phoneme structure:
    [[token, [ph1, ph2, ...]], ...]

Features
--------
- Chinese handled by pypinyin; optional tone numbers.
- Latin languages handled by MFA dicts (fallback to letter-by-letter if OOV).
- Mixed-language dispatcher splits the input sentence into zh / latin / punct segments.
- Optional BOS/EOS tokens and (disabled) phone separators.

CLI Example
-----------
python text2phoneme_mixed_fixed.py \
    --text "你好 world, 北京!" \
    --latin_lang english --use_tone --add_eos_bos

Dependencies
------------
- jieba
- pypinyin

Make sure your MFA dict paths below are valid.
"""

import re
import jieba
import argparse
from typing import List, Tuple, Dict
from pypinyin import pinyin, Style

########################
# Config & Registries  #
########################

PUNCS = '!,.?;:'
REGISTERED_TEXT_PROCESSORS: Dict[str, "BaseTxtProcessor"] = {}


def is_sil_phoneme(p: str) -> bool:
    return p == '' or not p or not p[0].isalpha()


def register_txt_processors(name):
    def _f(cls):
        REGISTERED_TEXT_PROCESSORS[name] = cls
        return cls
    return _f


def get_txt_processor_cls(name):
    return REGISTERED_TEXT_PROCESSORS.get(name, None)


#############################
# Base & Utility Processors #
#############################

class BaseTxtProcessor:
    """Base class to unify output format."""

    @staticmethod
    def sp_phonemes() -> List[str]:
        return ['|']

    @classmethod
    def process(cls, txt: str, preprocess_args: dict):
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct: List[List], preprocess_args: dict):
        # remove leading sil-phone tokens if any
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]

        # keep tail sil (e.g. punctuation) by design
        if preprocess_args.get('with_phsep', False):
            txt_struct = cls.add_bdr(txt_struct)

        if preprocess_args.get('add_eos_bos', False):
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct: List[List]):
        # placeholder: disabled separator injection to keep compatibility
        return txt_struct


########################################
# Chinese Processor (pypinyin + jieba) #
########################################

@register_txt_processors("zh")
class ChineseTxtProcessor(BaseTxtProcessor):
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @staticmethod
    def sp_phonemes() -> List[str]:
        return ['|', '#']

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.translate(ChineseTxtProcessor.table)
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(r"\s+", r"", text)
        return text

    @classmethod
    def pinyin_with_en(cls, txt: str, style) -> List[str]:
        """For each *character* return exactly one element.
        - Chinese char: its pinyin of given style
        - Latin/number: 'ENG' (will be handled elsewhere ideally)
        - Punctuation: the char itself
        """
        res = []
        for ch in txt:
            if re.match(r"[A-Za-z0-9]", ch):
                res.append('ENG')
            elif ch in PUNCS:
                res.append(ch)
            else:
                py = pinyin(ch, style=style, strict=False)[0][0]
                res.append(py)
        return res

    @classmethod
    def process(cls, txt: str, pre_args: dict):
        # Expect only Chinese + punctuation (mixed English should be split before reaching here)
        txt = cls.preprocess_text(txt)
        txt = txt.replace("嗯", "蒽")  # pypinyin fix

        use_tone = pre_args.get('use_tone', False)

        shengmu = cls.pinyin_with_en(txt, style=Style.INITIALS)
        yunmu = cls.pinyin_with_en(txt, style=Style.FINALS_TONE3 if use_tone else Style.FINALS)
        assert len(shengmu) == len(yunmu), (
            f"Lens mismatch: {len(shengmu)} vs {len(yunmu)} for text: {txt}")

        # Build phone list aligned to characters
        raw_ph_list = []
        for sm, ym, ch in zip(shengmu, yunmu, txt):
            if ch in PUNCS:
                raw_ph_list.append(ch)
                continue
            if sm == 'ENG' or ym == 'ENG':
                raw_ph_list.append('ENG')
                continue
            if sm == ym:
                raw_ph_list.append(sm)
            else:
                parts = [p for p in (sm, ym) if p]
                raw_ph_list.append('%'.join(parts) if parts else '')

        # jieba segmentation to insert '#'
        seg_list = '#'.join(jieba.cut(txt))
        # Count non-# chars in seg_list
        non_hash = [s for s in seg_list if s != '#']
        assert len(raw_ph_list) == len(non_hash), (raw_ph_list, seg_list)

        ph_list = []
        seg_idx = 0
        for p in raw_ph_list:
            if seg_list[seg_idx] == '#':
                ph_list.append('#')
                seg_idx += 1
            elif len(ph_list) > 0:
                ph_list.append('|')
            seg_idx += 1
            ph_list.extend([x for x in p.split('%') if x])

        # remove '#' around sil symbols safely
        sil_phonemes = list(PUNCS) + cls.sp_phonemes()
        cleaned = []
        for i in range(len(ph_list)):
            token = ph_list[i]
            if token == '#':
                prev_tok = ph_list[i-1] if i-1 >= 0 else None
                next_tok = ph_list[i+1] if i+1 < len(ph_list) else None
                if (prev_tok in sil_phonemes) or (next_tok in sil_phonemes):
                    continue
            cleaned.append(token)
        ph_list = cleaned

        # Map phones to per-char structure
        txt_struct = [[w, []] for w in txt]
        idx = 0
        for ph in ph_list:
            if ph in ['|', '#']:
                idx += 1
                continue
            if ph in [',', '.', '?', '!', ':']:
                if idx < len(txt_struct):
                    idx += 1
                if idx < len(txt_struct):
                    txt_struct[idx][1].append(ph)
                    idx += 1
                continue
            if idx < len(txt_struct):
                txt_struct[idx][1].append(ph)

        return cls.postprocess(txt_struct, pre_args), txt


#################################
# Latin languages via MFA dicts #
#################################

LANG_ENUM = {
    'english': 0,
    'german': 1,
    'russian': 2,
    'french': 3,
    'spanish': 4,
    'italian': 5,
    'japanese': 6,
    'korean': 7,
}

# Update these to your local paths
ENG_DICT_PATH = './mfa_dict/english_mfa.dict'
GER_DICT_PATH = './mfa_dict/german_mfa.dict'
RUS_DICT_PATH = './mfa_dict/russian_mfa.dict'
FRE_DICT_PATH = './mfa_dict/french.dict'
SPA_DICT_PATH = './mfa_dict/spanish_mfa.dict'
ITA_DICT_PATH = './mfa_dict/italian.dict'
JPN_DICT_PATH = './mfa_dict/japanese.dict'
KOR_DICT_PATH = './mfa_dict/korean_mfa.dict'

DICT_PATH_LIST = [
    ENG_DICT_PATH, GER_DICT_PATH, RUS_DICT_PATH, FRE_DICT_PATH,
    SPA_DICT_PATH, ITA_DICT_PATH, JPN_DICT_PATH, KOR_DICT_PATH
]


def load_dict(path: str) -> Dict[str, List[str]]:
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0].lower()
            phonemes = [p for p in parts[1:] if not p.replace('.', '', 1).isdigit()]
            d[word] = phonemes
    return d


_MFA_CACHE: Dict[int, Dict[str, List[str]]] = {}


def fetch_phonemes(word: str, language: int = 0) -> List[str]:
    if language not in _MFA_CACHE:
        _MFA_CACHE[language] = load_dict(DICT_PATH_LIST[language])
    return _MFA_CACHE[language].get(word.lower())


@register_txt_processors("latin")
class LatinTxtProcessor(BaseTxtProcessor):
    @classmethod
    def process(cls, txt: str, preprocess_args: dict):
        # split by space-ish: keep punctuation tokens
        tokens = re.findall(r"[A-Za-z']+|[0-9]+|[!,.?;:]", txt)

        lang_name = preprocess_args.get('latin_lang', 'english')
        lang_id = LANG_ENUM[lang_name]

        txt_struct = []
        for tk in tokens:
            if re.match(r"[!,.?;:]", tk):
                txt_struct.append([tk, [tk]])
            else:
                phs = fetch_phonemes(tk, lang_id)
                if phs is None:
                    phs = list(tk.upper())  # fallback
                txt_struct.append([tk, phs])

        return cls.postprocess(txt_struct, preprocess_args), txt


#############################
# Mixed-language dispatcher #
#############################

def split_mixed_sentence(text: str) -> List[Tuple[str, str]]:
    """Split text into segments tagged by 'zh', 'latin', or 'punct'."""

    segments: List[Tuple[str, str]] = []
    buff = ""
    mode = None

    def flush():
        nonlocal buff, mode
        if buff:
            segments.append((buff, mode))
            buff = ""

    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':  # Chinese
            if mode != 'zh':
                flush()
                mode = 'zh'
            buff += ch
        elif re.match(r"[A-Za-z0-9']", ch):  # Latin or number
            if mode != 'latin':
                flush()
                mode = 'latin'
            buff += ch
        elif ch in PUNCS or ch.isspace():
            flush()
            segments.append((ch, 'punct'))
            mode = None
        else:
            flush()
            segments.append((ch, 'punct'))
            mode = None
    flush()

    # merge adjacent punct
    merged: List[Tuple[str, str]] = []
    for s, t in segments:
        if merged and t == 'punct' and merged[-1][1] == 'punct':
            merged[-1] = (merged[-1][0] + s, 'punct')
        else:
            merged.append((s, t))
    return merged


def process_mixed(text: str,
                  zh_args: dict,
                  latin_args: dict) -> Tuple[List[List], str]:
    segments = split_mixed_sentence(text)
    out_struct: List[List] = []
    for seg, typ in segments:
        if typ == 'zh':
            zh_cls = get_txt_processor_cls("zh")
            struct, _ = zh_cls.process(seg, zh_args.copy())
            out_struct.extend(struct)
        elif typ == 'latin':
            lat_cls = get_txt_processor_cls("latin")
            struct, _ = lat_cls.process(seg, latin_args.copy())
            out_struct.extend(struct)
        else:  # punct
            for p in seg:
                if p.strip() == '':
                    continue
                out_struct.append([p, [p]])
    return out_struct, text

def get_phone(txt_struct):
    phs = [p for w in txt_struct for p in w[1]]
    ph_gb_word = ["_".join(w[1]) for w in txt_struct]
    words = [w[0] for w in txt_struct]
    ph2word = [w_id for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
    ph = " ".join(phs) 
    word = " ".join(words)
    ph_gb_word = " ".join(ph_gb_word)
    ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                    for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])

    return words, phs, ph2word, ph_gb_word_nosil

##################
#       CLI      #
##################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed Chinese/Latin text to phonemes")
    parser.add_argument("--text", type=str, required=True, help="Input sentence")
    parser.add_argument("--latin_lang", type=str, default="english", choices=list(LANG_ENUM.keys()))
    parser.add_argument("--use_tone", action="store_true", help="Use tone numbers for Chinese finals")
    parser.add_argument("--with_phsep", action="store_true", help="Insert phone separators (disabled internally)")
    parser.add_argument("--add_eos_bos", action="store_true", help="Add <BOS>/<EOS> tokens to the output")
    args = parser.parse_args()

    zh_args = {
        'use_tone': args.use_tone,
        'with_phsep': args.with_phsep,
        'add_eos_bos': False
    }
    latin_args = {
        'latin_lang': args.latin_lang,
        'with_phsep': args.with_phsep,
        'add_eos_bos': False
    }

    struct, raw = process_mixed(args.text, zh_args, latin_args)

    if args.add_eos_bos:
        struct = [["<BOS>", ["<BOS>"]]] + struct + [["<EOS>", ["<EOS>"]]]
    from pprint import pprint
    print("struct:")
    pprint(struct)

    words, phs, ph2word, ph_gb_word_nosil = get_phone(struct)
    print("words: ", words)
    print("phs: ", phs)
    print("ph2word: ", ph2word)
    print("ph_gb_word_nosil: ", ph_gb_word_nosil)
