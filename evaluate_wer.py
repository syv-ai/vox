#!/usr/bin/env python3
"""
Standalone WER evaluation script for Canary models.

This script can evaluate both pretrained and fine-tuned Canary models
on Danish speech recognition tasks and provide detailed error analysis.

Usage:
    # Evaluate pretrained model
    python evaluate_wer.py --model nvidia/canary-1b-v2 --manifest test_manifest.json

    # Evaluate fine-tuned model
    python evaluate_wer.py --model path/to/finetuned_model.nemo --manifest test_manifest.json

    # Compare two models
    python evaluate_wer.py --baseline nvidia/canary-1b-v2 --finetuned path/to/model.nemo --manifest test_manifest.json
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# NeMo and evaluation imports
try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
    from nemo.collections.asr.metrics.wer import WER
    import jiwer
except ImportError as e:
    print(f"Error importing required dependencies: {e}")
    print("Please install with: pip install nemo-toolkit[asr] jiwer")
    sys.exit(1)

# Audio processing
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Error importing audio libraries. Install with: pip install librosa soundfile")
    sys.exit(1)


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_model(model_path: str, device: Optional[str] = None) -> EncDecMultiTaskModel:
    """
    Load Canary model from path or hub.
    
    Args:
        model_path: Path to .nemo file or HuggingFace model name
        device: Device to load model on
        
    Returns:
        Loaded Canary model
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'cpu'  # Load on CPU first for MPS
        else:
            device = 'cpu'
    
    logging.info(f"Loading model: {model_path}")
    logging.info(f"Using device: {device}")
    
    try:
        if model_path.endswith('.nemo') or os.path.exists(model_path):
            # Load from local checkpoint
            model = EncDecMultiTaskModel.restore_from(
                restore_path=model_path,
                map_location=device
            )
        else:
            # Load from hub
            model = EncDecMultiTaskModel.from_pretrained(
                model_path,
                map_location=device
            )
        
        # Move to MPS if needed
        if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model = model.to('mps')
        
        model.eval()
        logging.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        raise


def load_manifest_data(manifest_path: str) -> List[Dict]:
    """Load manifest file and return list of entries."""
    logging.info(f"Loading manifest: {manifest_path}")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                if not entry.get('audio_filepath') or not entry.get('text'):
                    logging.warning(f"Line {line_num}: Missing required fields")
                    continue
                manifest_data.append(entry)
            except json.JSONDecodeError as e:
                logging.warning(f"Line {line_num}: JSON decode error: {e}")
                continue
    
    logging.info(f"Loaded {len(manifest_data)} valid entries from manifest")
    return manifest_data


def transcribe_batch(
    model: EncDecMultiTaskModel,
    audio_files: List[str],
    batch_size: int = 16,
    source_lang: str = 'da',
    target_lang: str = 'da'
) -> List[str]:
    """
    Transcribe a batch of audio files.
    
    Args:
        model: Canary model
        audio_files: List of audio file paths
        batch_size: Batch size for inference
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        List of transcription strings
    """
    transcriptions = []
    
    # Process in batches
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        
        try:
            # Transcribe batch
            batch_transcriptions = model.transcribe(
                paths2audio_files=batch_files,
                batch_size=len(batch_files),
                source_lang=source_lang,
                target_lang=target_lang,
                pnc=True,
                return_hypotheses=False
            )
            
            # Extract text from results
            for trans in batch_transcriptions:
                if hasattr(trans, 'text'):
                    transcriptions.append(trans.text)
                else:
                    transcriptions.append(str(trans))
                    
        except Exception as e:
            logging.error(f"Error transcribing batch {i//batch_size + 1}: {e}")
            # Add empty strings for failed batch
            transcriptions.extend([''] * len(batch_files))
    
    return transcriptions


def calculate_detailed_wer(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate detailed WER metrics including per-sample analysis.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        
    Returns:
        Dictionary with detailed WER metrics
    """
    # Overall WER using jiwer
    wer_overall = jiwer.wer(references, hypotheses) * 100
    
    # Per-sample WER calculation
    sample_wers = []
    sample_details = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        try:
            # Individual WER
            individual_wer = jiwer.wer([ref], [hyp]) * 100
            sample_wers.append(individual_wer)
            
            # Detailed alignment
            alignment = jiwer.compute_measures(ref, hyp)
            
            sample_details.append({
                'sample_id': i,
                'reference': ref,
                'hypothesis': hyp,
                'wer': individual_wer,
                'substitutions': alignment['substitutions'],
                'deletions': alignment['deletions'], 
                'insertions': alignment['insertions'],
                'hits': alignment['hits']
            })
            
        except Exception as e:
            logging.warning(f"Error calculating WER for sample {i}: {e}")
            sample_wers.append(100.0)  # Max error
            sample_details.append({
                'sample_id': i,
                'reference': ref,
                'hypothesis': hyp,
                'wer': 100.0,
                'substitutions': 0,
                'deletions': 0,
                'insertions': 0,
                'hits': 0
            })
    
    # Calculate statistics
    sample_wers = np.array(sample_wers)
    
    results = {
        'overall_wer': wer_overall,
        'mean_wer': np.mean(sample_wers),
        'median_wer': np.median(sample_wers),
        'std_wer': np.std(sample_wers),
        'min_wer': np.min(sample_wers),
        'max_wer': np.max(sample_wers),
        'perfect_matches': np.sum(sample_wers == 0),
        'total_samples': len(sample_wers),
        'sample_details': sample_details
    }
    
    return results


def analyze_errors(sample_details: List[Dict]) -> Dict:
    """
    Analyze common error patterns in the predictions.
    
    Args:
        sample_details: List of per-sample WER details
        
    Returns:
        Dictionary with error analysis
    """
    # Collect error types
    total_substitutions = sum(s['substitutions'] for s in sample_details)
    total_deletions = sum(s['deletions'] for s in sample_details)  
    total_insertions = sum(s['insertions'] for s in sample_details)
    total_hits = sum(s['hits'] for s in sample_details)
    total_words = total_hits + total_substitutions + total_deletions
    
    # Find samples with highest WER
    worst_samples = sorted(sample_details, key=lambda x: x['wer'], reverse=True)[:10]
    
    # Find common words that get confused
    word_errors = Counter()
    
    for sample in sample_details:
        ref_words = sample['reference'].lower().split()
        hyp_words = sample['hypothesis'].lower().split()
        
        # Simple alignment for word-level errors (approximation)
        if len(ref_words) == len(hyp_words):
            for ref_word, hyp_word in zip(ref_words, hyp_words):
                if ref_word != hyp_word:
                    word_errors[(ref_word, hyp_word)] += 1
    
    most_common_errors = word_errors.most_common(20)
    
    analysis = {
        'error_distribution': {
            'substitutions': total_substitutions,
            'deletions': total_deletions,
            'insertions': total_insertions,
            'hits': total_hits,
            'substitution_rate': total_substitutions / total_words * 100 if total_words > 0 else 0,
            'deletion_rate': total_deletions / total_words * 100 if total_words > 0 else 0,
            'insertion_rate': total_insertions / (total_hits + total_substitutions + total_insertions) * 100 if (total_hits + total_substitutions + total_insertions) > 0 else 0
        },
        'worst_samples': worst_samples,
        'common_word_errors': most_common_errors
    }
    
    return analysis


def save_detailed_results(results: Dict, output_path: str):
    """Save detailed results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Deep convert the results
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    results_converted = deep_convert(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Detailed results saved to: {output_path}")


def print_results_summary(results: Dict, model_name: str):
    """Print a summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"WER Evaluation Results - {model_name}")
    print(f"{'='*60}")
    print(f"Overall WER: {results['overall_wer']:.2f}%")
    print(f"Mean WER: {results['mean_wer']:.2f}%")
    print(f"Median WER: {results['median_wer']:.2f}%")
    print(f"WER Std: {results['std_wer']:.2f}%")
    print(f"WER Range: {results['min_wer']:.2f}% - {results['max_wer']:.2f}%")
    print(f"Perfect Matches: {results['perfect_matches']}/{results['total_samples']} ({results['perfect_matches']/results['total_samples']*100:.1f}%)")
    
    # Error analysis
    if 'error_analysis' in results:
        error_dist = results['error_analysis']['error_distribution']
        print(f"\nError Distribution:")
        print(f"  Substitutions: {error_dist['substitutions']} ({error_dist['substitution_rate']:.1f}%)")
        print(f"  Deletions: {error_dist['deletions']} ({error_dist['deletion_rate']:.1f}%)")
        print(f"  Insertions: {error_dist['insertions']} ({error_dist['insertion_rate']:.1f}%)")
        
        # Show worst samples
        print(f"\nWorst Performing Samples:")
        for i, sample in enumerate(results['error_analysis']['worst_samples'][:5]):
            print(f"  {i+1}. WER: {sample['wer']:.1f}%")
            print(f"     Ref: {sample['reference'][:80]}...")
            print(f"     Hyp: {sample['hypothesis'][:80]}...")
    
    print(f"{'='*60}\n")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate Canary model WER on Danish speech recognition")
    
    parser.add_argument(
        "--model", 
        type=str,
        help="Path to model (.nemo file) or HuggingFace model name"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline model for comparison (e.g., nvidia/canary-1b-v2)"
    )
    parser.add_argument(
        "--finetuned",
        type=str, 
        help="Fine-tuned model path for comparison"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to test manifest file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="da",
        help="Source language code"
    )
    parser.add_argument(
        "--target_lang", 
        type=str,
        default="da",
        help="Target language code"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction details to CSV file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not (args.baseline and args.finetuned):
        parser.error("Either --model or both --baseline and --finetuned must be specified")
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "evaluation.log")
    setup_logging(log_file)
    
    logging.info("Starting WER evaluation")
    logging.info(f"Arguments: {vars(args)}")
    
    # Load manifest data
    manifest_data = load_manifest_data(args.manifest)
    
    if args.max_samples:
        manifest_data = manifest_data[:args.max_samples]
        logging.info(f"Limited evaluation to {len(manifest_data)} samples")
    
    # Extract references and audio files
    references = [entry['text'].lower().strip() for entry in manifest_data]
    audio_files = [entry['audio_filepath'] for entry in manifest_data]
    
    # Check audio files exist
    missing_files = [f for f in audio_files if not os.path.exists(f)]
    if missing_files:
        logging.warning(f"{len(missing_files)} audio files not found")
        if len(missing_files) > 10:
            logging.warning(f"First 10 missing files: {missing_files[:10]}")
        else:
            logging.warning(f"Missing files: {missing_files}")
    
    results_summary = {}
    
    # Single model evaluation
    if args.model:
        logging.info("Evaluating single model")
        
        model = load_model(args.model)
        
        # Transcribe
        logging.info("Starting transcription...")
        hypotheses = transcribe_batch(
            model, audio_files, args.batch_size, args.source_lang, args.target_lang
        )
        
        # Normalize hypotheses
        hypotheses = [h.lower().strip() for h in hypotheses]
        
        # Calculate WER
        logging.info("Calculating WER metrics...")
        wer_results = calculate_detailed_wer(references, hypotheses)
        
        # Error analysis
        error_analysis = analyze_errors(wer_results['sample_details'])
        wer_results['error_analysis'] = error_analysis
        
        results_summary['single_model'] = {
            'model_path': args.model,
            'results': wer_results
        }
        
        # Print results
        print_results_summary(wer_results, args.model)
        
        # Save detailed results
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        save_detailed_results(wer_results, output_file)
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_df = pd.DataFrame({
                'reference': references,
                'hypothesis': hypotheses,
                'wer': [s['wer'] for s in wer_results['sample_details']],
                'audio_file': audio_files
            })
            predictions_file = os.path.join(args.output_dir, "predictions.csv")
            predictions_df.to_csv(predictions_file, index=False, encoding='utf-8')
            logging.info(f"Predictions saved to: {predictions_file}")
    
    # Comparison evaluation
    elif args.baseline and args.finetuned:
        logging.info("Evaluating model comparison")
        
        # Load both models
        baseline_model = load_model(args.baseline)
        finetuned_model = load_model(args.finetuned)
        
        # Evaluate baseline
        logging.info("Evaluating baseline model...")
        baseline_hypotheses = transcribe_batch(
            baseline_model, audio_files, args.batch_size, args.source_lang, args.target_lang
        )
        baseline_hypotheses = [h.lower().strip() for h in baseline_hypotheses]
        baseline_results = calculate_detailed_wer(references, baseline_hypotheses)
        baseline_results['error_analysis'] = analyze_errors(baseline_results['sample_details'])
        
        # Evaluate fine-tuned
        logging.info("Evaluating fine-tuned model...")
        finetuned_hypotheses = transcribe_batch(
            finetuned_model, audio_files, args.batch_size, args.source_lang, args.target_lang
        )
        finetuned_hypotheses = [h.lower().strip() for h in finetuned_hypotheses]
        finetuned_results = calculate_detailed_wer(references, finetuned_hypotheses)
        finetuned_results['error_analysis'] = analyze_errors(finetuned_results['sample_details'])
        
        # Calculate improvement
        wer_improvement = baseline_results['overall_wer'] - finetuned_results['overall_wer']
        relative_improvement = wer_improvement / baseline_results['overall_wer'] * 100
        
        results_summary['comparison'] = {
            'baseline_model': args.baseline,
            'finetuned_model': args.finetuned,
            'baseline_results': baseline_results,
            'finetuned_results': finetuned_results,
            'improvement': {
                'absolute': wer_improvement,
                'relative': relative_improvement
            }
        }
        
        # Print comparison results
        print_results_summary(baseline_results, f"Baseline ({args.baseline})")
        print_results_summary(finetuned_results, f"Fine-tuned ({args.finetuned})")
        
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Baseline WER: {baseline_results['overall_wer']:.2f}%")
        print(f"Fine-tuned WER: {finetuned_results['overall_wer']:.2f}%")
        print(f"Absolute improvement: {wer_improvement:.2f} percentage points")
        print(f"Relative improvement: {relative_improvement:.1f}%")
        print(f"{'='*60}\n")
        
        # Save comparison results
        output_file = os.path.join(args.output_dir, "comparison_results.json")
        save_detailed_results(results_summary['comparison'], output_file)
        
        # Save predictions comparison if requested
        if args.save_predictions:
            comparison_df = pd.DataFrame({
                'reference': references,
                'baseline_hypothesis': baseline_hypotheses,
                'finetuned_hypothesis': finetuned_hypotheses,
                'baseline_wer': [s['wer'] for s in baseline_results['sample_details']],
                'finetuned_wer': [s['wer'] for s in finetuned_results['sample_details']],
                'audio_file': audio_files
            })
            comparison_file = os.path.join(args.output_dir, "comparison_predictions.csv")
            comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
            logging.info(f"Comparison predictions saved to: {comparison_file}")
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()