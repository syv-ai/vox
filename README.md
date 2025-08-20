# Canary-1b-v2 Danish Finetuning

This repository contains a complete pipeline for finetuning NVIDIA's Canary-1b-v2 model, with fixes for tokenizer compatibility and model configuration issues.

## Features

- **Fixed Canary-1b-v2 compatibility**: Handles unified tokenizer vs concatenated tokenizer differences
- **Automatic tokenizer extraction**: Extracts and configures tokenizers from pretrained model
- **Vocab size mismatch handling**: Handles 16384→16400 token vocabulary differences
- **Position embedding fixes**: Corrects sequence length mismatches (512→1024)
- **Special tokens support**: Properly transfers special tokens like `<|startofcontext|>`

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies (if not already installed)
pip install nemo-toolkit[all]
pip install soundfile librosa
```

### 2. Run Training

```bash
python scripts/speech_to_text_aed.py \
    model.train_ds.manifest_filepath=datasets/LibriLight/train_manifest.json \
    model.validation_ds.manifest_filepath=datasets/LibriLight/train_manifest.json \
    model.test_ds.manifest_filepath=datasets/LibriLight/train_manifest.json \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.create_tensorboard_logger=false \
    trainer.max_steps=100 \
    trainer.log_every_n_steps=10
```

## What's Fixed

### Original Issues
- `RuntimeError: size mismatch for transf_decoder._embedding.token_embedding.weight`
- `KeyError: '<|startofcontext|>'` during data loading
- Position embedding size mismatch (512 vs 1024)

### Solutions Implemented
1. **Unified Tokenizer Detection**: Automatically detects Canary-1b-v2's unified tokenizer approach
2. **Dynamic Configuration**: Copies model architecture and tokenizer settings from pretrained model
3. **Manual Weight Loading**: Handles vocab size differences with custom weight initialization
4. **Special Token Transfer**: Ensures all special tokens are available for prompt formatting

## Key Files

- `scripts/speech_to_text_aed.py` - Main training script with all fixes
- `conf/speech_multitask/fast-conformer_aed.yaml` - Base configuration (dynamically modified)
- `datasets/LibriLight/train_manifest.json` - Training data manifest

## Training Process

The script automatically:
1. Downloads and loads Canary-1b-v2 pretrained model
2. Extracts tokenizers with all special tokens
3. Configures unified tokenizer for Canary-1b-v2
4. Handles vocabulary size mismatches during weight loading
5. Starts training with proper configuration

## Model Architecture

- **Encoder**: 810M parameters (ConformerEncoder)  
- **Decoder**: 151M parameters (TransformerDecoderNM)
- **Total**: 962M trainable parameters
- **Tokenizer**: Unified SentencePiece (16384 tokens + special tokens)

## Requirements

- NeMo Toolkit
- PyTorch with CUDA support
- LibriSound dataset (or your custom dataset)
- 16GB+ GPU memory recommended

## Output

The training produces:
- Model checkpoints in `nemo_experiments/default/checkpoints/`
- Training logs with step-by-step progress
- Prediction samples during validation

## Troubleshooting

If you encounter the original errors:
- Ensure you're using the updated `scripts/speech_to_text_aed.py`
- Verify `init_from_pretrained_model: nvidia/canary-1b-v2` is set
- Check that tokenizers are extracted to `./tokenizers/` directory