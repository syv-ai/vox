"""
Data preparation utilities for Canary Danish finetuning.
Converts HuggingFace datasets to NeMo manifest format.
"""

import os
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import datasets
from pathlib import Path



def ensure_mono_16khz(audio_array: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Convert audio to mono 16kHz format required by Canary.
    
    Args:
        audio_array: Input audio array
        sample_rate: Original sample rate
        
    Returns:
        Tuple of (processed_audio, 16000)
    """
    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)
    
    # Ensure float32
    audio_array = audio_array.astype(np.float32)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(
            audio_array, 
            orig_sr=sample_rate, 
            target_sr=16000
        )
        sample_rate = 16000
    
    # Normalize audio
    audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return audio_array, sample_rate


def create_manifest_entry(
    audio_filepath: str,
    text: str,
    duration: float,
    source_lang: str = "da",
    target_lang: str = "da"
) -> Dict:
    """
    Create a single manifest entry for NeMo training.
    
    Args:
        audio_filepath: Path to audio file
        text: Transcription text
        duration: Audio duration in seconds
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Dictionary containing manifest entry
    """
    return {
        "audio_filepath": audio_filepath,
        "text": text,
        "duration": duration,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "lang": target_lang,  # Some configs expect this field
        "pnc": "True",  # Punctuation and capitalization
        "answer": text  # Some Canary configs expect this
    }


def process_huggingface_dataset(
    dataset_name: str,
    output_dir: str,
    audio_column: str = "audio",
    text_column: str = "text",
    max_duration: float = 30.0,
    min_duration: float = 0.5,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    validation_size: float = 0.1,
    test_size: float = 0.1
) -> Dict[str, str]:
    """
    Process HuggingFace dataset and convert to NeMo manifest format.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for processed files
        audio_column: Name of audio column in dataset
        text_column: Name of text column in dataset
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        train_split: Name of training split
        validation_split: Name of validation split (None to create from train)
        test_split: Name of test split (None to create from train)
        validation_size: Fraction for validation if creating from train
        test_size: Fraction for test if creating from train
        
    Returns:
        Dictionary mapping split names to manifest file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = datasets.load_dataset(dataset_name)
    
    manifest_paths = {}
    
    # Process each split
    splits_to_process = []
    
    if train_split in dataset:
        splits_to_process.append((train_split, dataset[train_split]))
    
    if validation_split and validation_split in dataset:
        splits_to_process.append((validation_split, dataset[validation_split]))
    
    if test_split and test_split in dataset:
        splits_to_process.append((test_split, dataset[test_split]))
    
    # If we don't have validation/test splits, create them from train
    if len(splits_to_process) == 1 and train_split in [s[0] for s in splits_to_process]:
        print("Creating train/validation/test splits from training data")
        train_data = dataset[train_split]
        
        # Calculate split sizes
        total_size = len(train_data)
        val_size = int(total_size * validation_size)
        test_size_actual = int(total_size * test_size)
        train_size = total_size - val_size - test_size_actual
        
        # Create splits
        train_data = train_data.shuffle(seed=42)
        splits_data = train_data.train_test_split(
            test_size=val_size + test_size_actual,
            seed=42
        )
        
        val_test_data = splits_data['test'].train_test_split(
            test_size=test_size_actual,
            seed=42
        )
        
        splits_to_process = [
            ("train", splits_data['train']),
            ("validation", val_test_data['train']),
            ("test", val_test_data['test'])
        ]
    
    # Process each split
    for split_name, split_data in splits_to_process:
        print(f"\nProcessing {split_name} split ({len(split_data)} samples)")
        
        manifest_path = output_dir / f"{split_name}_manifest.json"
        manifest_paths[split_name] = str(manifest_path)
        
        valid_samples = 0
        skipped_samples = 0
        
        with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                try:
                    # Get audio and text
                    audio_data = sample[audio_column]
                    text = sample[text_column]
                    
                    if not text or not text.strip():
                        skipped_samples += 1
                        continue
                    
                    # Process audio
                    audio_array = np.array(audio_data['array'])
                    sample_rate = audio_data['sampling_rate']
                    
                    # Check duration
                    duration = len(audio_array) / sample_rate
                    if duration < min_duration or duration > max_duration:
                        skipped_samples += 1
                        continue
                    
                    # Convert to required format
                    audio_array, sample_rate = ensure_mono_16khz(audio_array, sample_rate)
                    
                    # Save audio file
                    audio_filename = f"{split_name}_{idx:06d}.wav"
                    audio_filepath = audio_dir / audio_filename
                    
                    sf.write(str(audio_filepath), audio_array, sample_rate)
                    
                    # Create manifest entry
                    manifest_entry = create_manifest_entry(
                        audio_filepath=str(audio_filepath),
                        text=text,
                        duration=duration
                    )
                    
                    # Write to manifest
                    manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                    valid_samples += 1
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    skipped_samples += 1
                    continue
        
        print(f"{split_name}: {valid_samples} valid samples, {skipped_samples} skipped")
    
    # Create dataset info file
    info_path = output_dir / "dataset_info.json"
    dataset_info = {
        "dataset_name": dataset_name,
        "splits": list(manifest_paths.keys()),
        "manifest_paths": manifest_paths,
        "audio_dir": str(audio_dir),
        "preprocessing": {
            "sample_rate": 16000,
            "channels": 1,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "text_normalization": "danish_lowercase"
        }
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset processing complete!")
    print(f"Dataset info saved to: {info_path}")
    print("Manifest files:")
    for split, path in manifest_paths.items():
        print(f"  {split}: {path}")
    
    return manifest_paths


def validate_manifest(manifest_path: str) -> Dict:
    """
    Validate a manifest file and return statistics.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Dictionary containing validation statistics
    """
    stats = {
        "total_samples": 0,
        "total_duration": 0.0,
        "min_duration": float('inf'),
        "max_duration": 0.0,
        "missing_files": 0,
        "text_lengths": [],
        "valid_samples": 0
    }
    
    print(f"Validating manifest: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                stats["total_samples"] += 1
                
                # Check required fields
                required_fields = ["audio_filepath", "text", "duration"]
                if not all(field in entry for field in required_fields):
                    print(f"Line {line_num}: Missing required fields")
                    continue
                
                # Check if audio file exists
                if not os.path.exists(entry["audio_filepath"]):
                    stats["missing_files"] += 1
                    continue
                
                # Update statistics
                duration = entry["duration"]
                stats["total_duration"] += duration
                stats["min_duration"] = min(stats["min_duration"], duration)
                stats["max_duration"] = max(stats["max_duration"], duration)
                stats["text_lengths"].append(len(entry["text"]))
                stats["valid_samples"] += 1
                
            except Exception as e:
                print(f"Line {line_num}: Error parsing JSON - {e}")
                continue
    
    # Calculate additional statistics
    if stats["text_lengths"]:
        stats["avg_text_length"] = np.mean(stats["text_lengths"])
        stats["median_text_length"] = np.median(stats["text_lengths"])
    
    if stats["valid_samples"] > 0:
        stats["avg_duration"] = stats["total_duration"] / stats["valid_samples"]
    
    print(f"Validation complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid samples: {stats['valid_samples']}")
    print(f"  Missing files: {stats['missing_files']}")
    print(f"  Total duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/3600:.2f} hours)")
    print(f"  Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
    if stats["valid_samples"] > 0:
        print(f"  Average duration: {stats['avg_duration']:.2f} seconds")
        print(f"  Average text length: {stats.get('avg_text_length', 0):.1f} characters")
    
    return stats


if __name__ == "__main__":
    # Example usage
    dataset_name = "alexandrainst/nota"
    output_dir = "./processed_data"
    
    # Process the dataset
    manifest_paths = process_huggingface_dataset(
        dataset_name=dataset_name,
        output_dir=output_dir,
        max_duration=30.0,
        min_duration=0.5
    )
    
    # Validate manifests
    for split, path in manifest_paths.items():
        validate_manifest(path)