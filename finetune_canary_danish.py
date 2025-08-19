#!/usr/bin/env python3
"""
Fine-tune Canary-1b-v2 for Danish speech recognition.

This script:
1. Downloads and preprocesses HuggingFace datasets
2. Evaluates baseline WER on the pretrained model
3. Fine-tunes the model on Danish data
4. Evaluates post-training WER
5. Saves the fine-tuned model

Usage:
    python finetune_canary_danish.py --dataset alexandrainst/nota --output_dir ./results
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict
import hydra
from hydra.core.global_hydra import GlobalHydra

# NeMo imports
try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
    from nemo.core.config import hydra_runner
    from nemo.utils import logging as nemo_logging
    from nemo.utils.exp_manager import exp_manager
    from nemo.collections.asr.metrics.wer import word_error_rate
    import pytorch_lightning as pl
except ImportError as e:
    print(f"Error importing NeMo dependencies: {e}")
    print("Please install NeMo with: pip install nemo-toolkit[asr]")
    sys.exit(1)

# Local imports
try:
    from utils.data_prep import process_huggingface_dataset, validate_manifest
except ImportError:
    print("Error importing local utilities. Make sure utils/data_prep.py exists.")
    sys.exit(1)


def setup_logging(output_dir: str, level: str = "INFO"):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set NeMo logging level
    nemo_logging.setLevel(getattr(logging, level.upper()))


def evaluate_wer(model: EncDecMultiTaskModel, manifest_path: str, batch_size: int = 16) -> float:
    """
    Evaluate Word Error Rate (WER) on a test set.
    
    Args:
        model: Trained Canary model
        manifest_path: Path to test manifest file
        batch_size: Batch size for inference
        
    Returns:
        WER as a percentage
    """
    logging.info(f"Evaluating WER on: {manifest_path}")
    
    # Transcribe the test set
    try:
        transcriptions = model.transcribe(
            paths2audio_files=[manifest_path],
            batch_size=batch_size,
            source_lang='da',
            target_lang='da',
            pnc=True,
            return_hypotheses=True
        )
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return float('inf')
    
    # Load ground truth from manifest
    ground_truths = []
    hypotheses = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_entries = [json.loads(line.strip()) for line in f]
    
    if len(transcriptions) != len(manifest_entries):
        logging.warning(f"Mismatch in lengths: {len(transcriptions)} transcriptions vs {len(manifest_entries)} manifest entries")
        min_len = min(len(transcriptions), len(manifest_entries))
        transcriptions = transcriptions[:min_len]
        manifest_entries = manifest_entries[:min_len]
    
    for trans, entry in zip(transcriptions, manifest_entries):
        if hasattr(trans, 'text'):
            hypothesis = trans.text
        else:
            hypothesis = str(trans)
        
        ground_truth = entry.get('text', '')
        
        hypotheses.append(hypothesis.lower().strip())
        ground_truths.append(ground_truth.lower().strip())
    
    # Calculate WER
    wer_value = word_error_rate(hypotheses=hypotheses, references=ground_truths)
    
    logging.info(f"WER: {wer_value:.2%}")
    logging.info(f"Sample predictions:")
    for i in range(min(5, len(hypotheses))):
        logging.info(f"  GT: {ground_truths[i]}")
        logging.info(f"  Pred: {hypotheses[i]}")
        logging.info(f"  ---")
    
    return wer_value * 100  # Return as percentage


def load_and_setup_model(model_name: str = "nvidia/canary-1b-v2") -> EncDecMultiTaskModel:
    """Load and setup the pretrained Canary model."""
    logging.info(f"Loading model: {model_name}")
    
    # Determine device
    if torch.cuda.is_available():
        map_location = 'cuda'
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        map_location = 'cpu'  # Load on CPU first, then move to MPS
        device = 'mps'
    else:
        map_location = 'cpu'
        device = 'cpu'
    
    logging.info(f"Using device: {device}")
    
    try:
        # Load the model
        model = EncDecMultiTaskModel.from_pretrained(
            model_name, 
            map_location=map_location
        )
        
        # Move to appropriate device
        if device == 'mps':
            model = model.to('mps')
        
        model.eval()
        logging.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def setup_training_config(
    config_path: str,
    manifest_paths: Dict[str, str],
    output_dir: str,
    model: EncDecMultiTaskModel
) -> OmegaConf:
    """Setup training configuration with dataset paths and model settings."""
    
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Update manifest paths
    with open_dict(cfg):
        cfg.model.train_ds.manifest_filepath = manifest_paths["train"]
        cfg.model.validation_ds.manifest_filepath = manifest_paths["validation"] 
        cfg.model.test_ds.manifest_filepath = manifest_paths["test"]
        
        # Set output directory
        cfg.exp_manager.exp_dir = output_dir
        
        # Copy model configuration from pretrained model
        cfg.model.model_defaults = model._cfg.get('model_defaults', {})
        cfg.model.preprocessor = model._cfg.preprocessor
        cfg.model.encoder = model._cfg.encoder
        cfg.model.transf_decoder = model._cfg.transf_decoder
        cfg.model.transf_encoder = model._cfg.get('transf_encoder', {})
        
        # Set prompt format and defaults from pretrained model
        cfg.model.prompt_format = model._cfg.prompt_format
        cfg.model.prompt_defaults = model._cfg.get('prompt_defaults', cfg.model.prompt_defaults)
        
        # Tokenizer configuration from pretrained model
        if not hasattr(cfg.model, 'tokenizer'):
            cfg.model.tokenizer = {}
        if not hasattr(cfg.model.tokenizer, 'langs'):
            cfg.model.tokenizer.langs = {}
            
        # Save tokenizers from pretrained model
        tokenizer_dir = os.path.join(output_dir, 'tokenizers')
        os.makedirs(tokenizer_dir, exist_ok=True)
        model.save_tokenizers(tokenizer_dir)
        
        # Update tokenizer paths in config
        for lang_dir in os.listdir(tokenizer_dir):
            lang_path = os.path.join(tokenizer_dir, lang_dir)
            if os.path.isdir(lang_path):
                if lang_dir not in cfg.model.tokenizer.langs:
                    cfg.model.tokenizer.langs[lang_dir] = {}
                cfg.model.tokenizer.langs[lang_dir].dir = lang_path
                cfg.model.tokenizer.langs[lang_dir].type = 'bpe'
        
        # Set special tokens directory
        cfg.spl_tokens.model_dir = os.path.join(tokenizer_dir, "spl_tokens")
        
        # Initialize from pretrained model
        cfg.init_from_pretrained_model = {
            'model0': {
                'name': "nvidia/canary-1b-v2"
            }
        }
        
        # Adjust batch sizes based on available memory
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 12:  # Less than 12GB VRAM
                cfg.model.train_ds.batch_size = 4
                cfg.model.validation_ds.batch_size = 8
                cfg.trainer.accumulate_grad_batches = 2
            elif gpu_memory_gb < 24:  # Less than 24GB VRAM
                cfg.model.train_ds.batch_size = 8
                cfg.model.validation_ds.batch_size = 16
        else:
            # CPU or MPS - use smaller batches
            cfg.model.train_ds.batch_size = 2
            cfg.model.validation_ds.batch_size = 4
            cfg.trainer.accumulate_grad_batches = 4
    
    logging.info(f"Training config setup complete")
    logging.info(f"Train batch size: {cfg.model.train_ds.batch_size}")
    logging.info(f"Validation batch size: {cfg.model.validation_ds.batch_size}")
    
    return cfg


@hydra_runner(config_path="config", config_name="danish_finetune")
def train_model(cfg: OmegaConf) -> str:
    """Train the Canary model with Hydra configuration."""
    
    # Setup experiment manager
    exp_dir = exp_manager(cfg.trainer, cfg.exp_manager)
    
    # Create model for training
    model = EncDecMultiTaskModel(cfg=cfg.model, trainer=cfg.trainer)
    
    # Setup trainer
    trainer = pl.Trainer(**cfg.trainer, logger=False)  # We'll add custom loggers if needed
    
    # Train the model
    logging.info("Starting training...")
    trainer.fit(model)
    
    # Save the final model
    final_model_path = os.path.join(exp_dir, "final_model.nemo")
    model.save_to(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    
    return final_model_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tune Canary-1b-v2 for Danish speech recognition")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="CoRal-project/coral-tts",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Output directory for results and model"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/danish_finetune.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--max_duration", 
        type=float, 
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--min_duration", 
        type=float, 
        default=0.5,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--skip_preprocessing", 
        action="store_true",
        help="Skip dataset preprocessing (use existing processed data)"
    )
    parser.add_argument(
        "--skip_baseline", 
        action="store_true",
        help="Skip baseline WER evaluation"
    )
    parser.add_argument(
        "--evaluate_only", 
        action="store_true",
        help="Only evaluate existing model, don't train"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to trained model for evaluation"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir, args.log_level)
    
    logging.info("Starting Canary Danish finetuning pipeline")
    logging.info(f"Arguments: {vars(args)}")
    
    # Step 1: Process dataset (if not skipping)
    processed_data_dir = os.path.join(args.output_dir, "processed_data")
    
    if not args.skip_preprocessing:
        logging.info("Step 1: Processing HuggingFace dataset")
        
        try:
            manifest_paths = process_huggingface_dataset(
                dataset_name=args.dataset,
                output_dir=processed_data_dir,
                max_duration=args.max_duration,
                min_duration=args.min_duration
            )
            
            # Validate manifests
            for split, path in manifest_paths.items():
                logging.info(f"Validating {split} manifest...")
                validate_manifest(path)
                
        except Exception as e:
            logging.error(f"Error processing dataset: {e}")
            sys.exit(1)
    else:
        # Load existing manifest paths
        info_path = os.path.join(processed_data_dir, "dataset_info.json")
        if not os.path.exists(info_path):
            logging.error(f"Dataset info not found at {info_path}. Please run without --skip_preprocessing first.")
            sys.exit(1)
            
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
        manifest_paths = dataset_info['manifest_paths']
        logging.info("Using existing processed dataset")
    
    # Step 2: Load pretrained model
    logging.info("Step 2: Loading pretrained Canary-1b-v2 model")
    try:
        pretrained_model = load_and_setup_model("nvidia/canary-1b-v2")
    except Exception as e:
        logging.error(f"Error loading pretrained model: {e}")
        sys.exit(1)
    
    # Step 3: Baseline evaluation (if not skipping)
    baseline_wer = None
    if not args.skip_baseline and "test" in manifest_paths:
        logging.info("Step 3: Evaluating baseline WER")
        try:
            baseline_wer = evaluate_wer(pretrained_model, manifest_paths["test"])
            logging.info(f"Baseline WER: {baseline_wer:.2f}%")
        except Exception as e:
            logging.error(f"Error evaluating baseline WER: {e}")
            baseline_wer = None
    
    # Step 4: Training (if not evaluation only)
    trained_model_path = None
    if not args.evaluate_only:
        logging.info("Step 4: Setting up training configuration")
        
        try:
            # Setup training config
            training_cfg = setup_training_config(
                config_path=args.config,
                manifest_paths=manifest_paths,
                output_dir=args.output_dir,
                model=pretrained_model
            )
            
            # Save updated config
            config_save_path = os.path.join(args.output_dir, "training_config.yaml")
            OmegaConf.save(training_cfg, config_save_path)
            logging.info(f"Training configuration saved to: {config_save_path}")
            
            # Clear Hydra instance if exists
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize Hydra with our config
            with hydra.initialize_config_dir(config_dir=str(Path(config_save_path).parent.absolute())):
                with hydra.compose(config_name="training_config"):
                    logging.info("Starting model training...")
                    trained_model_path = train_model(training_cfg)
                    
        except Exception as e:
            logging.error(f"Error during training: {e}")
            import traceback
            logging.error(traceback.format_exc())
            sys.exit(1)
    
    # Step 5: Post-training evaluation
    final_wer = None
    if args.evaluate_only and args.model_path:
        model_to_evaluate_path = args.model_path
    elif trained_model_path:
        model_to_evaluate_path = trained_model_path
    else:
        model_to_evaluate_path = None
    
    if model_to_evaluate_path and "test" in manifest_paths:
        logging.info("Step 5: Evaluating trained model")
        try:
            # Load trained model
            trained_model = EncDecMultiTaskModel.restore_from(model_to_evaluate_path)
            final_wer = evaluate_wer(trained_model, manifest_paths["test"])
            logging.info(f"Final WER: {final_wer:.2f}%")
            
        except Exception as e:
            logging.error(f"Error evaluating trained model: {e}")
            final_wer = None
    
    # Step 6: Save results summary
    results = {
        "dataset": args.dataset,
        "baseline_wer": baseline_wer,
        "final_wer": final_wer,
        "improvement": None,
        "trained_model_path": trained_model_path,
        "config": args.config,
        "processing_args": {
            "max_duration": args.max_duration,
            "min_duration": args.min_duration
        }
    }
    
    if baseline_wer is not None and final_wer is not None:
        results["improvement"] = baseline_wer - final_wer
        results["relative_improvement"] = (baseline_wer - final_wer) / baseline_wer * 100
    
    results_path = os.path.join(args.output_dir, "results_summary.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Training pipeline completed!")
    logging.info(f"Results summary saved to: {results_path}")
    
    if baseline_wer is not None:
        logging.info(f"Baseline WER: {baseline_wer:.2f}%")
    if final_wer is not None:
        logging.info(f"Final WER: {final_wer:.2f}%")
    if results.get("improvement") is not None:
        logging.info(f"WER improvement: {results['improvement']:.2f} percentage points")
        logging.info(f"Relative improvement: {results['relative_improvement']:.1f}%")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    main()