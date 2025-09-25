#!/usr/bin/env python3
# Fine-tune Mistral 7B for hurricane impact forecasting using PEFT/LoRA

import json, argparse, math, re, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Import our custom modules
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle
from summary_report import generate_all_reports
from visual_plot import plot_best_worst_pois

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-input", help="Training JSONL file")
    p.add_argument("--test-input", help="Test JSONL file") 
    p.add_argument("--val-input", help="Validation JSONL file")
    p.add_argument("--outdir", required=True, help="Output directory for fine-tuned model")
    p.add_argument("--city", required=True, help="City name for this fine-tuning run")
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    p.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    p.add_argument("--limit", type=int, help="Limit number of samples for testing")
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    p.add_argument("--resume-from", help="Resume training from checkpoint")
    p.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing model")
    p.add_argument("--generate-plots", action="store_true", help="Generate performance plots")
    p.add_argument("--num-plots", type=int, default=10, help="Number of best/worst plots to generate")
    return p.parse_args()

def create_training_prompt(prev13, loc, cat, series_start, landfall, target_date, y_true):
    """Create training prompt with ground truth"""
    values = ", ".join(f"{float(v):.1f}" for v in prev13)
    return (
        "You are an expert forecaster. Given 13 daily visit counts, predict day 14. "
        "Reply ONLY: 'PREDICTION: <number>'.\n"
        f"Location: {loc}\nCategory: {cat}\n"
        f"SeriesStart: {series_start}\n"
        f"Landfall: {landfall}\n"
        f"TargetDate: {target_date}\n"
        f"Visits(1-13): {values}\n"
        f"PREDICTION: {y_true:.1f}"
    )

def create_inference_prompt(prev13, loc, cat, series_start, landfall, target_date):
    """Create inference prompt without ground truth"""
    values = ", ".join(f"{float(v):.1f}" for v in prev13)
    return (
        "You are an expert forecaster. Given 13 daily visit counts, predict day 14. "
        "Reply ONLY: 'PREDICTION: <number>'.\n"
        f"Location: {loc}\nCategory: {cat}\n"
        f"SeriesStart: {series_start}\n"
        f"Landfall: {landfall}\n"
        f"TargetDate: {target_date}\n"
        f"Visits(1-13): {values}\n"
        "PREDICTION: "
    )

def load_jsonl_data(jsonl_path, limit=None):
    """Load data from a single JSONL file"""
    rows = []
    with open(jsonl_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No records in JSONL file: {jsonl_path}")
    
    # Apply limit if specified
    if limit and limit > 0:
        df = df.head(limit)
        print(f"Using subset: {len(df)} samples (limited from {len(rows)})")
    
    return df

def create_prompts_from_df(df):
    """Create training prompts from dataframe"""
    prompts = []
    for _, row in df.iterrows():
        prompt = create_training_prompt(
            row["prev_13_values"], 
            row.get("location_name", ""), 
            row.get("top_category", ""),
            row["series_start_date"], 
            row["landfall_date"], 
            row["target_date_d14"],
            row["y_true_d14"]
        )
        prompts.append(prompt)
    return prompts

def load_and_prepare_data(train_jsonl_path=None, test_jsonl_path=None, val_jsonl_path=None, limit=None):
    """Load pre-split training, validation, and test data"""
    
    # Handle pre-split data mode
    print("Loading pre-split training, validation, and test data...")
    
    # Check if all three files are provided
    if train_jsonl_path and val_jsonl_path and test_jsonl_path:
        print("Using all three pre-split files:")
        print(f"  Training: {train_jsonl_path}")
        print(f"  Validation: {val_jsonl_path}")
        print(f"  Test: {test_jsonl_path}")
        
        # Load all three datasets
        train_df = load_jsonl_data(train_jsonl_path, limit)
        val_df = load_jsonl_data(val_jsonl_path)
        test_df = load_jsonl_data(test_jsonl_path)
        
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        print(f"Loaded {len(test_df)} test samples")
        
    else:
        raise ValueError("Must provide all three files: --train-input, --val-input, and --test-input")
    
    # Create prompts for all three sets
    train_texts = create_prompts_from_df(train_df)
    val_texts = create_prompts_from_df(val_df)
    test_texts = create_prompts_from_df(test_df)
    
    print(f"Final split sizes:")
    print(f"  Training: {len(train_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    return train_texts, val_texts, test_texts, test_df

def create_datasets(train_texts, val_texts, test_texts, tokenizer, max_length):
    """Create tokenized datasets for training, validation, and testing"""
    def tokenize_function(examples):
        # Ensure we're working with a list of strings
        texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
        
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors=None,  # Return lists, not tensors
            add_special_tokens=True
        )
        
        # Copy input_ids to labels for causal language modeling
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized
    
    # Ensure all texts are strings
    train_texts = [str(text) for text in train_texts]
    val_texts = [str(text) for text in val_texts]
    test_texts = [str(text) for text in test_texts]
    
    print(f"Sample training text: {train_texts[0][:200]}...")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing training data"
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing validation data"
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing test data"
    )
    
    return train_dataset, val_dataset, test_dataset

def setup_model_and_tokenizer(model_name, use_8bit=False):
    """Setup quantized model and tokenizer with LoRA"""
    
    # Quantization config
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer

def setup_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):  # More conservative settings
    """Setup LoRA configuration for Mistral"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def parse_prediction(text):
    """Parse prediction from model output with fallback handling"""
    text = text.strip()
    
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', text, re.I)
    
    if m:
        pred = float(m.group(1))
        pred = max(0.0, min(pred, 10000.0))
        return pred
    
    pred_match = re.search(r'PREDICTION[:\s]*([0-9]*\.?[0-9]+)', text, re.I)
    if pred_match:
        pred = max(0.0, min(float(pred_match.group(1)), 10000.0))
        return pred
    
    direct_match = re.search(r'^([0-9]*\.?[0-9]+)', text, re.I)
    if direct_match:
        pred = max(0.0, min(float(direct_match.group(1)), 10000.0))
        return pred
    
    nums = re.findall(r'([0-9]+\.?[0-9]*)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred
    
    float_nums = re.findall(r'([0-9]*\.?[0-9]+)', text)
    if float_nums:
        pred = max(0.0, min(float(float_nums[0]), 10000.0))
        return pred
    
    print(f"Warning: Could not parse prediction from: '{text[:100]}'")
    return None

def evaluate_model(model, tokenizer, test_df, outdir, city_name, generate_plots=False, num_plots=10):
    """Evaluate fine-tuned model on test set using custom modules"""
    from transformers import pipeline
    
    print(f"Evaluating on {len(test_df)} test samples...")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        do_sample=False,
        max_new_tokens=30,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    results = []
    y_true_list = []
    y_pred_list = []
    failed_predictions = 0
    
    for idx, row in test_df.iterrows():
        prompt = create_inference_prompt(
            row["prev_13_values"], 
            row.get("location_name", ""), 
            row.get("top_category", ""),
            row["series_start_date"], 
            row["landfall_date"], 
            row["target_date_d14"]
        )
        
        try:
            output = pipe(prompt, return_full_text=False)
            generated_text = output[0]["generated_text"].strip()
            
            if idx < 3:
                print(f"Sample {idx+1} generated: '{generated_text}'")
            
            pred = parse_prediction(generated_text)
            
            if pred is not None:
                y_true = row["y_true_d14"]
                y_true_list.append(y_true)
                y_pred_list.append(pred)
                
                abs_error = abs(y_true - pred)
                pct_error = (abs_error / max(y_true, 1)) * 100 if y_true != 0 else np.nan
                
                denominator = (abs(y_true) + abs(pred)) / 2
                if denominator != 0:
                    smape_ind = (abs(y_true - pred) / denominator) * 100
                else:
                    smape_ind = np.nan
                
                y_true_clip = max(y_true, 0)
                y_pred_clip = max(pred, 0)
                rmsle_ind = (np.log1p(y_pred_clip) - np.log1p(y_true_clip)) ** 2
                
                results.append({
                    "placekey": row["placekey"],
                    "location_name": row.get("location_name", ""),
                    "top_category": row.get("top_category", ""),
                    "city": row.get("city", ""),
                    "latitude": row.get("latitude", np.nan),
                    "longitude": row.get("longitude", np.nan),
                    "series_start_date": row["series_start_date"],
                    "landfall_date": row["landfall_date"],
                    "target_date_d14": row["target_date_d14"],
                    "actual_target_date": row.get("actual_target_date", ""),
                    "target_days_after_landfall": row.get("target_days_after_landfall", np.nan),
                    "time_periods_used": row.get("time_periods_used", ""),
                    "prev_13_values": row["prev_13_values"],
                    "y_true_d14": y_true,
                    "y_pred_d14": pred,
                    "absolute_error": abs_error,
                    "percent_error": pct_error,
                    "smape_individual": smape_ind,
                    "rmsle_individual": rmsle_ind,
                    "source_city": city_name,
                    "llm_text": generated_text[:200]
                })
            else:
                failed_predictions += 1
                if idx < 5:
                    print(f"Failed to parse prediction {idx+1}: '{generated_text}'")
                    
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            failed_predictions += 1
            continue
    
    print(f"Successfully parsed {len(results)}/{len(test_df)} predictions")
    if failed_predictions > 0:
        print(f"Failed to parse {failed_predictions} predictions")
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty and len(y_true_list) > 0:
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        mae = compute_mae(y_true_arr, y_pred_arr)
        rmse = compute_rmse(y_true_arr, y_pred_arr)
        smape_mean, smape_median = compute_smape(y_true_arr, y_pred_arr)
        rmsle = compute_rmsle(y_true_arr, y_pred_arr)
        
        overall_metrics = {
            'samples': len(y_true_list),
            'mae': mae,
            'rmse': rmse,
            'smape_mean': smape_mean,
            'smape_median': smape_median,
            'rmsle': rmsle
        }
        
        print(f"\n=== Fine-tuned Mistral Evaluation ===")
        print(f"Test samples: {overall_metrics['samples']}")
        print(f"MAE: {overall_metrics['mae']:.3f}")
        print(f"RMSE: {overall_metrics['rmse']:.3f}")
        print(f"sMAPE (mean): {overall_metrics['smape_mean']:.3f}%")
        print(f"sMAPE (median): {overall_metrics['smape_median']:.3f}%")
        print(f"RMSLE: {overall_metrics['rmsle']:.3f}")
        
        csv_path = outdir / "results" / "finetuned_predictions.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
        
        metrics_path = outdir / "results" / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(overall_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        try:
            print("\n=== Generating Comprehensive Reports ===")
            reports_dir = outdir / "reports"
            
            report_results = generate_all_reports(
                pred_df=results_df,
                output_dir=reports_dir,
                base_name=city_name.lower()
            )
            
            if report_results:
                print("Generated comprehensive reports:")
                for report_type, path in report_results['file_paths'].items():
                    print(f"  {report_type}: {path.name}")
            
        except Exception as e:
            print(f"Warning: Could not generate comprehensive reports: {e}")
        
        try:
            print("\n=== Generating Performance Visualization Plots ===")
            if generate_plots:
                plot_output_dir = outdir / "plots"
                plot_best_worst_pois(
                    csv_path=csv_path,
                    output_dir=plot_output_dir,
                    num_plots=num_plots,
                    show_confidence=False
                )
                print(f"Visualization plots saved to: {plot_output_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate visualization plots: {e}")
        
        if "top_category" in results_df.columns and len(results_df) > 0:
            try:
                category_metrics = results_df.groupby("top_category").agg({
                    "y_true_d14": "count",
                    "absolute_error": ["mean", "std"],
                    "smape_individual": "mean",
                    "rmsle_individual": "mean"
                }).round(3)
                
                category_metrics.columns = ["count", "mean_ae", "std_ae", "mean_smape", "mean_rmsle"]
                category_metrics = category_metrics.reset_index()
                category_path = outdir / "results" / "category_metrics.csv"
                category_metrics.to_csv(category_path, index=False)
                print(f"Category metrics saved to: {category_path}")
            except Exception as e:
                print(f"Warning: Could not generate category metrics: {e}")
        
        return overall_metrics
    else:
        print("No valid predictions generated - check model output format")
        if not results_df.empty:
            results_df.to_csv(outdir / "results" / "failed_predictions_debug.csv", index=False)
        return {}

def main():
    # Disable problematic integrations
    import os
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    args = parse_args()
    
    # Input validation - require all three files
    if not args.train_input or not args.val_input or not args.test_input:
        print("Error: Must provide all three files:")
        print("  --train-input: Training JSONL file")
        print("  --val-input: Validation JSONL file") 
        print("  --test-input: Test JSONL file")
        sys.exit(1)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Setup output directories
    results_root = Path("Result_mistral_output")
    results_root.mkdir(exist_ok=True)
    
    city_dir = results_root / args.city.lower().replace(" ", "_")
    city_dir.mkdir(exist_ok=True)
    
    run_name = f"mistral7b_run_{timestamp}"
    outdir = city_dir / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    (outdir / "model").mkdir(exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "results").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    
    print(f"=== Hurricane Forecasting Fine-tuning with Mistral ===")
    print(f"Model: {args.model}")
    print(f"City: {args.city}")
    print(f"Output: {outdir}")
    print(f"Directory structure:")
    print(f"  Results root: {results_root}")
    print(f"  City folder: {city_dir}")
    print(f"  Run folder: {run_name}")
    
    print("\n=== Loading Data ===")
    print("Using full pre-split mode (train/validation/test)")
    train_texts, val_texts, test_texts, test_df = load_and_prepare_data(
        args.train_input, args.test_input, args.val_input, args.limit
    )
    
    if args.evaluate_only:
        print("\n=== Evaluation Only Mode ===")
        model_path = outdir / "model" / "final_model"
        if not model_path.exists():
            print(f"Error: No model found at {model_path}")
            sys.exit(1)
            
        base_model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
        model = PeftModel.from_pretrained(base_model, model_path)
        evaluate_model(model, tokenizer, test_df, outdir, args.city, args.generate_plots, args.num_plots)
        return
    
    # Setup model and tokenizer
    print("\n=== Setting up Mistral Model ===")
    model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
    
    # Apply LoRA
    print("\n=== Applying LoRA ===")
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset, val_dataset, test_dataset = create_datasets(train_texts, val_texts, test_texts, tokenizer, args.max_length)
    
    # Training arguments - Updated to match Llama settings
    training_args = TrainingArguments(
        output_dir=outdir / "checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,  # Updated to match Llama
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,  # Added to match Llama
        max_grad_norm=1.0,  # Updated to match Llama
        logging_steps=25,  # Updated to match Llama
        logging_dir=outdir / "logs",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,  # Updated to match Llama
        save_total_limit=5,  # Updated to match Llama
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        resume_from_checkpoint=args.resume_from,
        dataloader_pin_memory=False,  # Added to match Llama
        bf16=True,  # Added to match Llama
        fp16=False,  # Added to match Llama
        remove_unused_columns=False,  # Added to match Llama
    )
    
    # Data collator - Updated to match Llama
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Added to match Llama
        return_tensors="pt"  # Added to match Llama
    )
    
    # Create trainer with validation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use validation dataset for monitoring
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n=== Starting Training ({args.epochs} epochs) ===")
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    print(f"Final testing on {len(test_dataset)} samples")
    
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from)
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = outdir / "model" / "final_model"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Model saved to: {final_model_path}")
    
    # Final evaluation on test set
    print(f"\n=== Evaluating Fine-tuned Mistral on Test Set ===")
    evaluation_metrics = evaluate_model(
        trainer.model, tokenizer, test_df, outdir, args.city, 
        args.generate_plots, args.num_plots
    )
    
    # Create training summary
    summary = {
        "city": args.city,
        "run_name": run_name,
        "timestamp": timestamp,
        "model": args.model,
        "training_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "test_samples": len(test_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "training_time_minutes": training_time / 60,
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
        "best_eval_loss": min([log.get("eval_loss", float('inf')) for log in trainer.state.log_history if "eval_loss" in log], default="N/A"),
        "evaluation_metrics": evaluation_metrics
    }
    
    with open(outdir / "results" / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {outdir / 'results' / 'training_summary.json'}")
    
    print(f"\n=== Final Summary ===")
    print(f"City: {args.city}")
    print(f"Run: {run_name}")
    print(f"Data split:")
    print(f"  Training: {len(train_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    print(f"Training completed successfully!")
    print(f"All outputs saved to: {outdir}")
    if evaluation_metrics:
        print(f"Final Test Results:")
        print(f"  MAE: {evaluation_metrics.get('mae', 'N/A')}")
        print(f"  RMSE: {evaluation_metrics.get('rmse', 'N/A')}")
        print(f"  sMAPE: {evaluation_metrics.get('smape_mean', 'N/A')}%")
        print(f"  RMSLE: {evaluation_metrics.get('rmsle', 'N/A')}")

if __name__ == "__main__":
    main()