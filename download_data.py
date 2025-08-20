import os
import tarfile
import wget
import glob
import tqdm
import librosa
import json
import numpy as np
import torch
import subprocess
import sys
import shutil
from omegaconf import OmegaConf


# DOWNLOAD DATA 
def download_and_prepare_librilight_data(data_dir="datasets"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    libri_data_dir = os.path.join(data_dir, 'LibriLight')
    libri_tgz_file = f'{data_dir}/librispeech_finetuning.tgz'

    if not os.path.exists(libri_tgz_file):
        url = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
        libri_path = wget.download(url, data_dir, bar=None)
        print(f"Dataset downloaded at: {libri_path}")

    if not os.path.exists(libri_data_dir):
        tar = tarfile.open(libri_tgz_file)
        tar.extractall(path=libri_data_dir)

    print(f'LibriLight data is ready at {libri_data_dir}')

download_and_prepare_librilight_data()


# INFERENCE - Removed IPython specific code


# BUILDING THE DATASET
def build_manifest(data_root, manifest_path):
    transcript_list = glob.glob(os.path.join(data_root, 'LibriLight/librispeech_finetuning/1h/**/*.txt'), recursive=True)
    tot_duration = 0
    with open(manifest_path, 'w') as fout:
        pass # make sure a new file is created
    for transcript_path in tqdm.tqdm(transcript_list):
        with open(transcript_path, 'r') as fin:
            wav_dir = os.path.dirname(transcript_path)
            with open(manifest_path, 'a') as fout:
                for line in fin:
                    # Lines look like this:
                    # fileID transcript
                    file_id = line.strip().split(' ')[0]
                    audio_path = os.path.join(wav_dir, f'{file_id}.flac')

                    transcript = ' '.join(line.strip().split(' ')[1:]).lower()
                    transcript = transcript.strip()

                    duration = librosa.core.get_duration(path=audio_path)
                    tot_duration += duration
                    # Write the metadata to the manifest
                    metadata = {
                      "audio_filepath": audio_path,
                      "duration": duration,
                      "text": transcript,
                      "lang": "en",
                      "target_lang": "en",
                      "source_lang": "en",
                      "pnc": "False"
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')
    print(f'\n{np.round(tot_duration/3600)} hour audio data ready for training')

data_dir = "datasets"
train_manifest = os.path.join(data_dir, 'LibriLight/train_manifest.json')
build_manifest(data_dir, train_manifest)
print(f"LibriLight train manifests created at {train_manifest}.")

# LOADING THE MODEL
from nemo.collections.asr.models import EncDecMultiTaskModel
map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2', map_location=map_location)
canary_model.prompt_format = "canary2"

# Use a simple configuration that relies on init_from_pretrained_model
# This avoids tokenizer conflicts by letting NeMo handle the initialization properly

base_model_cfg = OmegaConf.load("config/fast-conformer_aed.yaml")
base_model_cfg['name'] = None  # Set to None to avoid logger conflicts
base_model_cfg.pop("spl_tokens", None)  # Remove spl_tokens to use unified tokenizer
base_model_cfg.pop("init_from_nemo_model", None)  # Remove to avoid conflict with init_from_pretrained_model

# Set mandatory values that were ??? in the config
base_model_cfg['model']['prompt_format'] = "canary2"
base_model_cfg['model']['tokenizer']['langs']['en']['dir'] = "./tokenizers"
base_model_cfg['model']['tokenizer']['langs']['spl_tokens']['dir'] = "./tokenizers/spl_tokens"
base_model_cfg['model']['prompt_defaults'] = [
    {
        "role": "user",
        "slots": {
            "decodercontext": "",
            "source_lang": "<|en|>",
            "target_lang": "<|en|>",
            "emotion": "<|emo:undefined|>",
            "pnc": "<|pnc|>",
            "itn": "<|noitn|>",
            "diarize": "<|nodiarize|>",
            "timestamp": "<|notimestamp|>"
        }
    }
]

base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-1b-v2"

cfg = OmegaConf.create(base_model_cfg)
with open("config/canary-1b-v2-finetune.yaml", "w") as f:
    OmegaConf.save(cfg, f)

MANIFEST = os.path.join("datasets", "LibriLight", 'train_manifest.json')

# Create the conf directory structure that the training script expects
conf_dir = "conf/speech_multitask"
os.makedirs(conf_dir, exist_ok=True)

# Move our config to the expected location
shutil.copy("config/canary-1b-v2-finetune.yaml", f"{conf_dir}/fast-conformer_aed.yaml")

# Run the training script (note: no config path/name args since they're hardcoded in the decorator)
cmd = [
    sys.executable, "scripts/speech_to_text_aed.py",
    f"model.train_ds.manifest_filepath={MANIFEST}",
    f"model.validation_ds.manifest_filepath={MANIFEST}",
    f"model.test_ds.manifest_filepath={MANIFEST}",
    "exp_manager.resume_ignore_no_checkpoint=true",
    "exp_manager.create_tensorboard_logger=false",  # Disable to avoid logger conflict
    "trainer.max_steps=10",
    "trainer.log_every_n_steps=1"
]

env = os.environ.copy()
env['HYDRA_FULL_ERROR'] = '1'

result = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)