#!/usr/bin/env python3
"""
Simple script to split training data for multiple cities into train/validation files
Creates 70% training, 10% validation from original 80% training data for each city
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def split_training_data(input_train_file, output_dir, val_ratio=0.125):
    """Split training data into train/validation"""
    
    print(f"Loading training data from: {input_train_file}")
    rows = []
    with open(input_train_file, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    print(f"Total training samples: {len(df)}")
    
    # Split into train/validation
    train_df, val_df = train_test_split(
        df, 
        test_size=val_ratio, 
        random_state=42,
        shuffle=True
    )
    
    print(f"New training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save new training file
    train_output = output_dir / "train_70.jsonl"
    with open(train_output, "w") as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    
    # Save validation file
    val_output = output_dir / "validation_10.jsonl"
    with open(val_output, "w") as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    
    print(f"Files created:")
    print(f"  New training: {train_output}")
    print(f"  Validation: {val_output}")
    
    return train_output, val_output

def main():
    parser = argparse.ArgumentParser(description="Split training data for multiple cities")
    parser.add_argument("--train-files", nargs='+', 
                       help="Full paths to training files")
    parser.add_argument("--base-dir", default="prepared_data", 
                       help="Base directory containing city folders (used with --cities)")
    parser.add_argument("--cities", nargs='+', 
                       default=["tampa", "orlando", "miami", "cape coral"],
                       help="List of cities to process (used with --base-dir)")
    parser.add_argument("--val-ratio", type=float, default=0.125,
                       help="Validation ratio (0.125 = 12.5% for 70/10 split)")
    
    args = parser.parse_args()
    
    # Determine input method
    if args.train_files:
        print(f"Processing training files: {args.train_files}")
        train_files = [Path(f) for f in args.train_files]
    else:
        print(f"Processing cities: {args.cities}")
        print(f"Base directory: {args.base_dir}")
        train_files = []
        for city in args.cities:
            city_dir = Path(args.base_dir) / f"{city}_2025_09_01" / "data"
            train_file = city_dir / f"{city}_train.jsonl"
            train_files.append(train_file)
    
    print(f"Validation ratio: {args.val_ratio} ({args.val_ratio*100:.1f}%)")
    
    # Process all files
    successful = 0
    failed = 0
    
    for train_file in train_files:
        print(f"\n{'='*50}")
        print(f"Processing: {train_file}")
        print(f"{'='*50}")
        
        if not train_file.exists():
            print(f"Warning: Training file not found: {train_file}")
            failed += 1
            continue
        
        try:
            # Output to same directory as input file
            output_dir = train_file.parent
            split_training_data(train_file, output_dir, args.val_ratio)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {train_file}: {e}")
            failed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total: {successful} successful, {failed} failed")
    print(f"\nFiles created: train_70.jsonl and validation_10.jsonl in each city's data directory")

if __name__ == "__main__":
    main()