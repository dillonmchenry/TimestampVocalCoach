# STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation

This is the official PyTorch implementation of **[STARS]**, a unified framework for singing voice **transcription**, **alignment**, and **refined style annotation**. It includes the full implementation and pretrained model checkpoints.  
**This project builds upon the foundation laid by [ROSVOT](https://github.com/RickyL-2000/ROSVOT).**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.06670)

Visit our [demo page](https://gwx314.github.io/stars-demo/) for audio samples.

---

## What STARS Can Do

1. **Automatic Singing Alignment**  
   Aligns phoneme and word sequences to audio, producing precise timing for each phoneme.

2. **Automatic Singing Transcription**  
   Converts audio waveforms into MIDI-style note sequences.

3. **Singing Technique Prediction**  
   Predicts whether specific vocal techniques are present for each phoneme.

4. **Global Singing Style Classification**  
   Estimates global vocal style attributes such as emotion and rhythmic character.

---

## Dependencies

STARS is tested with **Python 3.10**, **PyTorch 2.4.0**, and **CUDA 12.8**.

```bash
conda create -n stars python=3.10
conda activate stars
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export PYTHONPATH=.
```

## Pretrained Model Checkpoints

Download the pretrained models from [HuggingFace](https://huggingface.co/verstar/STARS), save into the `checkpoints` directory. It includes:

| Model             | Description               | Path                          |
|------------------|---------------------------|-------------------------------|
| STARS (Chinese)  | Chinese Singing annotation model  | `checkpoints/stars_chinese`  |
| STARS (chinese_english)  | Chinese an English Singing annotation model  | `checkpoints/stars_bilingual`  |
| RMVPE             | Pitch extraction model    | `checkpoints/rmvpe`          |


## Inference

First, prepare the metadata in a standardized `metadata.json` format. Note that using pure vocal audio will yield better results. You can use Ultimate Vocal Remover or other tools for vocal-instrument separation. Each entry should be a dictionary with the following fields:

- `item_name`: Unique ID of the audio segment  
- `wav_fn`: Path to the waveform  
- `word`: List of word-level annotations  
- `ph`: List of phonemes  
- `ph2words`: Mapping from phonemes to corresponding word indices  
- Optionally:  
  - `word_durs`: Word duration list  
  - `ph_durs`: Phoneme duration list  
  If these durations are not provided, they will be predicted automatically.

You can process the Chinese singing audio:
```bash
python scripts/process_ch.py -i [path-to-wav-dir] -o [output-json-path]
```

Example metadata.json entry json:
```json
[
  {
    "item_name": "segment#1_0",
    "wav_fn": "input/cut_wavs/segment#1_0.wav",
    "word": ["晨", "雾", "飘", "渺", "听", "山", "泉", "谁", "在", "林", "间", "如", "泼", "墨", "画", "面"],
    "ph": ["ch", "en", "u", "p", "iao", "m", "iao", "t", "ing", "sh", "an", "q", "uan", "sh", "uei", "z", "ai", "l", "in", "j", "ian", "r", "u", "p", "o", "m", "o", "h", "ua", "m", "ian"],
    "ph2words": [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
  }
]
```

You can also get the phonemes of mixed Chinese and English text using the following command:
```bash
python scripts/mixedtext2phoneme.py --text "welcome to 北京" --with_phsep
```

Run Inference

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python inference/stars.py --ckpt [path-to-ckpt] --config [config-file] --phset [path-to-phset] -o [output-dir] --metadata [metadata.json]
```
Please note that the `--config` and `--phset` should correspond to the `--ckpt`. For instance, in the case of the bilingual model, the config needs to be set to `configs/stars_bilingual.yaml`, and the phset to `chinese_and_english_phone_set.json`.

Optional Flags
- `--save_plot`: Save visualizations for alignment and prediction results.
- `-v`: Enable verbose logging for detailed output.
- `--bsz`: Set the batch size per GPU (default is usually safe, adjust if out-of-memory).
- `--max_tokens`: Limit the number of tokens per batch to control memory usage.
- `--ds_workers`: Number of CPU worker processes for data preprocessing (helps speed up loading).


## Training

### Data Preparation

We provide a preprocessing pipeline using [GTSinger](https://github.com/AaronZ345/GTSinger) as an example. The same logic can be adapted for other datasets with similar annotations.

If the phoneme sequence is not fully accurate (e.g., one phoneme corresponds to multiple notes), use the following script to flatten and preprocess the phoneme structure:

```bash
python scripts/process_ph.py -i [input-json-path] -o [output-json-path]
```

Once you have the processed manifest file (e.g., `data/processed/chinese/metadata_processed.json`), you can binarize the dataset.

Before that, inspect the configuration file [configs/stars_chinese.yaml](configs/stars_chinese.yaml). The test_prefixes field specifies which samples to use for testing. Feel free to modify it based on your needs.

To binarize the dataset:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python data_gen/run.py --config configs/stars_chinese.yaml
```

Additionally, we need external noise datasets for robust training (you can disable noise injection by simply setting the `noise_prob` hyper-parameter in [configs/stars_chinese.yaml](configs/stars_chinese.yaml) to 0.0). We use [MUSAN](https://www.openslr.org/17/) dataset as the noise source. Once you download and unzip the dataset, replace the value of the `raw_data_dir` attribute in [configs/musan.yaml](configs/musan.yaml) with the current path of MUSAN, and run the following command to binarize the noise source:

```shell
python data_gen/run.py --config configs/musan.yaml
```

### Model Training

To train the model, run:

```shell
CUDA_VISIBLE_DEVICES=[your-gpus] python tasks/run.py --config configs/stars_chinese.yaml --exp_name [your-exp-name] --reset
```
- Replace `[your-exp-name]` with your desired experiment name.
- All checkpoints and logs will be saved in `checkpoints/[your-exp-name]/`.

> ⚠️ **Note**: If you are training on another language or using multilingual data, be sure to modify the `ph_num` field in the config file to match the total number of phonemes.


## Acknowledgements

This implementation uses parts of the code from the following GitHub repos:
[ROSVOT](https://github.com/RickyL-2000/ROSVOT),
[LyricAlignment](https://github.com/navi0105/LyricAlignment),
[SOFA](https://github.com/qiuqiao/SOFA).

## Citation

If you find this code useful in your research, please cite our work:

```bibtex
@article{guo2025stars,
  title={STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation},
  author={Guo, Wenxiang and Zhang, Yu and Pan, Changhao and Zhu, Zhiyuan and Li, Ruiqi and Chen, Zhetao and Xu, Wenhao and Wu, Fei and Zhao, Zhou},
  journal={arXiv preprint arXiv:2507.06670},
  year={2025}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech/singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.