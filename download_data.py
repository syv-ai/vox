import os
import tarfile

# Install dependencies
#!pip install wget
#!apt-get update && apt-get install -y sox libsndfile1 ffmpeg
#!pip install text-unidecode
#!pip install omegaconf

#BRANCH='main'

#!python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[asr]


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


# INFERENCE
from pydub import AudioSegment
from IPython.display import Audio, display

def listen_to_audio(audio_path, offset=0.0, duration=-1):
    audio = AudioSegment.from_file(audio_path)
    start_ms = int(offset * 1000)
    if duration == -1:
        end_ms = -1
    else:
        end_ms = int((offset+duration) * 1000)

    segment = audio[start_ms:end_ms]
    audio = Audio(segment.export(format='wav').read())
    display(audio)


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

# TRAINING FROM A CHECKPOINT
# Load canary model if not previously loaded in this notebook instance
if 'canary_model' not in locals():
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2')

base_model_cfg = OmegaConf.load("config/fast-conformer_aed.yaml")

base_model_cfg['name'] = 'canary-1b-v2-finetune'
base_model_cfg.pop("init_from_nemo_model", None)
base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-1b-v2"

canary_model.save_tokenizers("./tokenizers")

for lang in os.listdir('tokenizers'):
    base_model_cfg['model']['tokenizer']['langs'][lang] = {}
    base_model_cfg['model']['tokenizer']['langs'][lang]['dir'] = os.path.join('tokenizers', lang)
    base_model_cfg['model']['tokenizer']['langs'][lang]['type'] = 'bpe'
base_model_cfg['spl_tokens']['model_dir'] = os.path.join('tokenizers', "spl_tokens")

base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']

base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']

cfg = OmegaConf.create(base_model_cfg)
with open("config/canary-1b-v2-finetune.yaml", "w") as f:
    OmegaConf.save(cfg, f)

MANIFEST = os.path.join("datasets", "LibriLight", 'train_manifest.json')
!HYDRA_FULL_ERROR=1 python scripts/speech_to_text_aed.py \
  --config-path="../config" \
  --config-name="canary-180m-flash-finetune.yaml" \
  name="canary-180m-flash-finetune" \
  model.train_ds.manifest_filepath={MANIFEST} \
  model.validation_ds.manifest_filepath={MANIFEST} \
  model.test_ds.manifest_filepath={MANIFEST} \
  exp_manager.exp_dir="canary_results" \
  exp_manager.resume_ignore_no_checkpoint=true \
  trainer.max_steps=10 \
  trainer.log_every_n_steps=1