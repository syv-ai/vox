# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Training the model
```sh
python speech_to_text_aed.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.tarred_audio_filepaths=<path to tar files with audio> \
    model.train_ds.manifest_filepath=<path to audio data manifest> \
    model.train_ds.batch_duration=360 \
    model.train_ds.num_buckets=30 \
    model.train_ds.bucket_duration_bins=<optional list of precomputed float bins for bucket durations, speeds up init> \
    model.validation_ds.manifest_filepath=<path to validation manifest> \
    model.test_ds.manifest_filepath=<path to test manifest> \
    model.model_defaults.asr_enc_hidden=1024 \
    model.model_defaults.lm_enc_hidden=512 \
    model.model_defaults.lm_dec_hidden=1024 \
    model.tokenizer.langs.spl_tokens.dir=<path to the directory of prompt special tokens tokenizer> \
    model.tokenizer.langs.spl_tokens.type=bpe \
    model.tokenizer.langs.en.dir=<path to the directory of en language tokenizer (add new langs the same way)> \
    model.tokenizer.langs.en.type=bpe \
    model.prompt_format="canary" \
    trainer.devices=-1 \
    trainer.accelerator="ddp" \
    trainer.max_steps=100000 \
    +trainer.limit_train_batches=20000 \
    trainer.val_check_interval=5000 \
    +trainer.use_distributed_sampler=false \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```


"""
import lightning.pytorch as pl
import os
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(config_path="../conf/speech_multitask/", config_name="fast-conformer_aed")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Check for spl tokens to create spl_tokenizer.
    if cfg.get("spl_tokens"):
        logging.info("Detected spl_tokens config. Building tokenizer.")
        spl_cfg = cfg["spl_tokens"]
        spl_tokenizer_cls = model_utils.import_class_by_path(cfg.model.tokenizer.custom_tokenizer["_target_"])
        spl_tokenizer_cls.build_special_tokenizer(
            spl_cfg["tokens"], spl_cfg["model_dir"], force_rebuild=spl_cfg["force_rebuild"]
        )
        cfg.model.tokenizer.langs.spl_tokens.dir = spl_cfg["model_dir"]

    # Extract tokenizers from pretrained model if needed
    if cfg.get("init_from_pretrained_model"):
        logging.info(f"Extracting tokenizers from pretrained model: {cfg.init_from_pretrained_model}")
        
        # Load pretrained model temporarily to extract tokenizers
        pretrained_model = EncDecMultiTaskModel.from_pretrained(cfg.init_from_pretrained_model)
        
        # Copy model configuration from pretrained model
        cfg.model.model_defaults = pretrained_model._cfg.get('model_defaults', {})
        cfg.model.preprocessor = pretrained_model._cfg.preprocessor
        cfg.model.encoder = pretrained_model._cfg.encoder
        cfg.model.transf_decoder = pretrained_model._cfg.transf_decoder
        cfg.model.transf_encoder = pretrained_model._cfg.get('transf_encoder', {})
        
        # Set prompt format and defaults from pretrained model
        cfg.model.prompt_format = pretrained_model._cfg.prompt_format
        cfg.model.prompt_defaults = pretrained_model._cfg.get('prompt_defaults', cfg.model.prompt_defaults)
        
        # Detect tokenizer type: Canary-1b-v2 uses unified, others use concatenated
        is_unified_tokenizer = (
            cfg.init_from_pretrained_model == "nvidia/canary-1b-v2" or
            (hasattr(pretrained_model, 'tokenizer') and 
             hasattr(pretrained_model.tokenizer, 'tokenizer') and
             not hasattr(pretrained_model.tokenizer, 'tokenizers'))  # Single tokenizer, not multiple
        )
        
        logging.info(f"Detected tokenizer type: {'unified' if is_unified_tokenizer else 'concatenated'}")
        
        # Save tokenizers from pretrained model
        tokenizer_dir = os.path.join(os.getcwd(), 'tokenizers')
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        if is_unified_tokenizer:
            # For Canary-1b-v2: unified tokenizer approach
            logging.info("Configuring unified tokenizer for Canary-1b-v2")
            
            # Save the unified tokenizer
            pretrained_model.save_tokenizers(tokenizer_dir)
            
            # For unified tokenizer, use simpler configuration
            from omegaconf import DictConfig
            tokenizer_config = DictConfig({
                'dir': tokenizer_dir,
                'type': 'bpe',
                'model_path': os.path.join(tokenizer_dir, 'tokenizer.model')
            })
            
            # Copy other important tokenizer settings from pretrained model
            if hasattr(pretrained_model._cfg, 'tokenizer'):
                if hasattr(pretrained_model._cfg.tokenizer, 'custom_tokenizer'):
                    tokenizer_config['custom_tokenizer'] = pretrained_model._cfg.tokenizer.custom_tokenizer
                    
            cfg.model.tokenizer = tokenizer_config
                
        else:
            # For other Canary models: concatenated tokenizer approach
            logging.info("Configuring concatenated tokenizer")
            
            # Tokenizer configuration from pretrained model
            if not hasattr(cfg.model, 'tokenizer'):
                cfg.model.tokenizer = {}
            if not hasattr(cfg.model.tokenizer, 'langs'):
                cfg.model.tokenizer.langs = {}
            
            # Create language-specific directories
            en_tokenizer_dir = os.path.join(tokenizer_dir, 'en')
            os.makedirs(en_tokenizer_dir, exist_ok=True)
            
            pretrained_model.save_tokenizers(en_tokenizer_dir)
            
            # Update tokenizer paths in config to use the extracted tokenizers
            cfg.model.tokenizer.langs.en.dir = en_tokenizer_dir
            cfg.model.tokenizer.langs.en.type = 'bpe'
            
            # Handle special tokens if they exist in the pretrained model
            spl_tokens_dir = os.path.join(tokenizer_dir, "spl_tokens")
            if hasattr(pretrained_model, 'tokenizer') and hasattr(pretrained_model.tokenizer, 'tokenizers'):
                if 'spl_tokens' in pretrained_model.tokenizer.tokenizers:
                    os.makedirs(spl_tokens_dir, exist_ok=True)
                    # Copy spl_tokens tokenizer files if they exist
                    for file_name in ['tokenizer.model', 'tokenizer.vocab', 'vocab.txt']:
                        src_path = os.path.join(en_tokenizer_dir, file_name)
                        if os.path.exists(src_path):
                            import shutil
                            shutil.copy2(src_path, os.path.join(spl_tokens_dir, file_name))
                    
                    cfg.model.tokenizer.langs.spl_tokens.dir = spl_tokens_dir
                    cfg.model.tokenizer.langs.spl_tokens.type = 'bpe'
                    
                    if cfg.get("spl_tokens"):
                        cfg.spl_tokens.model_dir = spl_tokens_dir

        # Clean up pretrained model to free memory
        del pretrained_model
        
        logging.info(f"Tokenizers extracted and saved to: {tokenizer_dir}")

    aed_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    # Handle vocab size differences manually
    if cfg.get("init_from_pretrained_model"):
        logging.info("Loading pretrained checkpoint with manual handling of vocab size differences")
        restored_model = EncDecMultiTaskModel.from_pretrained(cfg.init_from_pretrained_model)
        
        # Get state dicts
        pretrained_state = restored_model.state_dict()
        current_state = aed_model.state_dict()
        
        # Copy weights that match exactly
        compatible_state = {}
        vocab_size_mismatches = []
        
        for key in current_state.keys():
            if key in pretrained_state:
                if current_state[key].shape == pretrained_state[key].shape:
                    # Shapes match exactly - copy directly
                    compatible_state[key] = pretrained_state[key]
                elif 'token_embedding.weight' in key or 'log_softmax.mlp.layer0' in key:
                    # Handle vocab size mismatch for embedding and output layers
                    pretrained_param = pretrained_state[key]
                    current_param = current_state[key]
                    
                    if 'token_embedding.weight' in key:
                        # Copy the first N tokens from pretrained (16384) to current model
                        min_vocab_size = min(pretrained_param.shape[0], current_param.shape[0])
                        compatible_state[key] = current_param.clone()
                        compatible_state[key][:min_vocab_size] = pretrained_param[:min_vocab_size]
                        vocab_size_mismatches.append(f"{key}: {pretrained_param.shape} -> {current_param.shape}")
                        
                    elif 'log_softmax.mlp.layer0.weight' in key:
                        # Copy weights for output layer
                        min_output_size = min(pretrained_param.shape[0], current_param.shape[0])
                        compatible_state[key] = current_param.clone()
                        compatible_state[key][:min_output_size] = pretrained_param[:min_output_size]
                        vocab_size_mismatches.append(f"{key}: {pretrained_param.shape} -> {current_param.shape}")
                        
                    elif 'log_softmax.mlp.layer0.bias' in key:
                        # Copy bias for output layer
                        min_output_size = min(pretrained_param.shape[0], current_param.shape[0])
                        compatible_state[key] = current_param.clone()
                        compatible_state[key][:min_output_size] = pretrained_param[:min_output_size]
                        vocab_size_mismatches.append(f"{key}: {pretrained_param.shape} -> {current_param.shape}")
                else:
                    logging.warning(f"Shape mismatch for {key}: {pretrained_state[key].shape} vs {current_state[key].shape}")
        
        # Load the compatible weights
        missing_keys, unexpected_keys = aed_model.load_state_dict(compatible_state, strict=False)
        
        if vocab_size_mismatches:
            logging.info(f"Successfully handled vocab size mismatches: {vocab_size_mismatches}")
        if missing_keys:
            logging.info(f"Missing keys during checkpoint loading: {missing_keys}")
        if unexpected_keys:
            logging.info(f"Unexpected keys during checkpoint loading: {unexpected_keys}")
            
        # Transfer special tokens from pretrained model to current model
        if hasattr(restored_model, 'tokenizer') and hasattr(aed_model, 'tokenizer'):
            if hasattr(restored_model.tokenizer, 'special_tokens'):
                logging.info("Transferring special tokens from pretrained model")
                aed_model.tokenizer.special_tokens = restored_model.tokenizer.special_tokens.copy()
                logging.info(f"Transferred {len(aed_model.tokenizer.special_tokens)} special tokens")
                
                # Verify critical special tokens are present
                critical_tokens = ['<|startofcontext|>', '<|endofcontext|>', '<|startoftranscript|>', '<|endoftranscript|>']
                missing_tokens = []
                for token in critical_tokens:
                    if token not in aed_model.tokenizer.special_tokens:
                        missing_tokens.append(token)
                
                if missing_tokens:
                    logging.warning(f"Missing critical special tokens: {missing_tokens}")
                else:
                    logging.info("All critical special tokens successfully transferred")
            
        del restored_model
    else:
        aed_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # Freeze most parameters to reduce memory usage for 24GB GPU
    logging.info("Freezing encoder and transformer encoder parameters for memory optimization")
    
    # It might be possible to invoke freeze_params directly in the config but I can't get it to work
    # the idea being that we add +model.freeze_params=['^encoder\\..*'] to the config (not sure about the regex or name)
    aed_model.encoder.freeze()

    trainer.fit(aed_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if aed_model.prepare_test(trainer):
            trainer.test(aed_model)


if __name__ == '__main__':
    main()