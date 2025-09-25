#!/usr/bin/env python3
"""
Simple RAG Dataset Augmentation - adds minimal evacuation context fields
"""

import json
import argparse
from pathlib import Path
import pandas as pd

class SimpleEvacuationKB:
    def __init__(self, kb_json_path):
        kb_path = Path(kb_json_path)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
        with open(kb_path, 'r', encoding='latin1') as f:
            self.kb_data = json.load(f)
        self.entries = self.kb_data.get('entries', [])

    def get_evacuation_info(self, fips_code, target_date):
        """Get simple evacuation info: severity_score and days_since_effective"""
        if not fips_code:
            return 0, 999, False
        
        evacuation_order_types = [
            "Mandatory evacuation order", 
            "Voluntary evacuation order"
        ]
        
        fips_orders = [e for e in self.entries 
                       if e.get('fips_code') == fips_code 
                       and e.get('order_type') in evacuation_order_types
                       and e.get('evacuation_area') 
                       and str(e.get('evacuation_area')).lower() not in ['nan', 'null', '', 'none']]
        
        if not fips_orders:
            return 0, 999, False
        
        target_dt = pd.to_datetime(target_date)
        active_orders = []
        
        for order in fips_orders:
            effective_date = order.get('effective_date') or order.get('announcement_date')
            if not effective_date:
                continue
                
            effective_dt = pd.to_datetime(effective_date)
            if effective_dt <= target_dt:
                days_since = (target_dt - effective_dt).days
                
                # Severity: 3=mandatory, 2=voluntary
                severity = 3 if "Mandatory" in order.get('order_type', '') else 2
                
                active_orders.append({
                    'severity': severity,
                    'days_since': days_since
                })
        
        if not active_orders:
            return 0, 999, False
        
        # Get most severe order, break ties by recency
        best = max(active_orders, key=lambda x: (x['severity'], -x['days_since']))
        
        # RAG activation: mandatory orders within 7 days
        rag_active = best['severity'] >= 3 and 0 <= best['days_since'] <= 7
        
        return best['severity'], best['days_since'], rag_active

def augment_dataset(input_file, output_file, kb_path):
    """Add simple RAG fields to dataset"""
    evac_kb = SimpleEvacuationKB(kb_path)
    
    print(f"Processing {input_file}...")
    
    augmented_records = []
    rag_activated = 0
    total = 0
    
    with open(input_file, 'r', encoding='latin1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            total += 1
            
            fips_code = record.get('fips_county_code')
            target_date = record.get('target_date_d14', record.get('actual_target_date'))
            
            severity, days_since, rag_flag = evac_kb.get_evacuation_info(fips_code, target_date)
            
            # Add just 3 simple fields
            record['evacuation_severity'] = severity
            record['evacuation_days_since'] = days_since  
            record['rag_active'] = rag_flag
            
            if rag_flag:
                rag_activated += 1
            
            augmented_records.append(record)
    
    # Save augmented dataset
    with open(output_file, 'w', encoding='latin1') as f:
        for record in augmented_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Augmentation complete:")
    print(f"  Total records: {total}")
    print(f"  RAG activated: {rag_activated} ({100*rag_activated/total:.1f}%)")
    print(f"  Saved to: {output_file}")

def process_multiple_files(input_files, kb_path, mode='augment'):
    """Process multiple files at once"""
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        # Generate output filename
        if mode == 'augment':
            output_file = str(input_path.parent / f"rag_{input_path.name}")
        else:
            output_file = str(input_path.parent / f"prompts_{input_path.name}")
        
        print(f"\n--- Processing {input_file} ---")
        
        if mode == 'augment':
            augment_dataset(input_file, output_file, kb_path)
        else:
            create_training_prompts(input_file, output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--input-files", nargs='+', help="Multiple input files")
    parser.add_argument("--input-file", help="Single input file") 
    parser.add_argument("--output-file", help="Single output file")
    parser.add_argument("--mode", choices=['augment', 'prompts'], default='augment',
                       help="augment=add fields, prompts=create training prompts")
    args = parser.parse_args()
    
    if args.input_files:
        # Process multiple files
        process_multiple_files(args.input_files, args.knowledge_base, args.mode)
    elif args.input_file and args.output_file:
        # Process single file
        if args.mode == 'augment':
            augment_dataset(args.input_file, args.output_file, args.knowledge_base)
        else:
            create_training_prompts(args.input_file, args.output_file)
    else:
        print("Error: Provide either --input-files for multiple files or --input-file + --output-file for single file")

if __name__ == "__main__":
    main()
