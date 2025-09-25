#!/usr/bin/env python3
# Run Llama 3 8B (quantized) on prepared 13->14 windows and save predictions.

import json, argparse, math, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Import the additional modules
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle
from summary_report import generate_all_reports
from visual_plot import plot_best_worst_pois

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Prepared JSONL from data_prep.py")
    p.add_argument("--output", required=True, help="final_predictions.csv")
    p.add_argument("--city", required=True, help="City name for output directory organization")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    p.add_argument("--generate-reports", action="store_true", help="Generate comprehensive reports and visualizations")
    p.add_argument("--num-plots", type=int, default=10, help="Number of best/worst plots to generate")
    return p.parse_args()

def create_prompt(prev13, loc, cat, series_start, landfall, target_date):
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

def parse_pred(text, fallback):
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', text, re.I)
    if m:
        pred = float(m.group(1))
        pred = max(0.0, min(pred, 10000.0))
        return pred
    # fallback: first number
    nums = re.findall(r'([0-9]*\.?[0-9]+)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred
    return float(fallback)

def calculate_individual_smape(y_true, y_pred):
    """Calculate sMAPE for a single prediction"""
    denominator = (abs(y_true) + abs(y_pred)) / 2
    if denominator != 0:
        return (abs(y_true - y_pred) / denominator) * 100
    else:
        return np.nan

def calculate_individual_rmsle(y_true, y_pred):
    """Calculate RMSLE for a single prediction"""
    y_true_clipped = max(y_true, 0)
    y_pred_clipped = max(y_pred, 0)
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    return (log_pred - log_true) ** 2

def load_prepared(jsonl_path):
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    df = load_prepared(args.input)
    if df.empty:
        print("[ERROR] No records in prepared JSONL.")
        sys.exit(1)

    # Quantized load
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb = BitsAndBytesConfig(load_in_8bit=args.use_8bit) if args.use_8bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True, quantization_config=bnb)

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )

    # Warm-up
    _ = gen("Warm up. PREDICTION:", max_new_tokens=4)

    # Build prompts
    prompts, meta = [], []
    for i, r in df.iterrows():
        prompts.append(create_prompt(
            r["prev_13_values"], r.get("location_name", ""), r.get("top_category",""),
            r["series_start_date"], r["landfall_date"], r["target_date_d14"]
        ))
        meta.append((r["placekey"], r.get("city",""), r.get("latitude", np.nan), r.get("longitude", np.nan),
                     r.get("location_name",""), r.get("top_category",""), r["y_true_d14"],
                     r["prev_13_values"], r["series_start_date"], r["landfall_date"], r["target_date_d14"],
                     r.get("actual_target_date", r["target_date_d14"]), r.get("target_days_after_landfall", None),
                     r.get("time_periods_used","")))

    # Batched inference
    out_rows = []
    bs = max(1, args.batch_size)
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        batch = prompts[s:s+bs]
        outs = gen(batch, return_full_text=False)
        for (pk, city, lat, lon, name, cat, y_true, prev13, s_start, lfall, tgt_d, act_tgt, days_after, tp_used), o in zip(meta[s:s+bs], outs):
            text = (o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]).strip()
            pred = parse_pred(text, prev13[-1] if prev13 else 0.0)
            abs_err = abs(y_true - pred)
            pct_err = (abs_err / max(y_true, 1)) * 100 if y_true != 0 else np.nan
            
            # Calculate individual sMAPE and RMSLE for each prediction
            smape_individual = calculate_individual_smape(y_true, pred)
            rmsle_individual = calculate_individual_rmsle(y_true, pred)
            
            out_rows.append({
                "placekey": pk, "city": city, "latitude": lat, "longitude": lon,
                "location_name": name, "top_category": cat,
                "series_start_date": s_start, "landfall_date": lfall,
                "target_date_d14": tgt_d, "actual_target_date": act_tgt,
                "target_days_after_landfall": days_after, "time_periods_used": tp_used,
                "y_true_d14": y_true, "y_pred_d14": pred,
                "absolute_error": abs_err, "percent_error": pct_err,
                "smape_individual": smape_individual, "rmsle_individual": rmsle_individual,
                "prev_13_values": prev13,
                "llm_text": text[:200]
            })

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(out_rows)

    # Calculate comprehensive metrics using imported functions
    valid = pred_df.dropna(subset=["y_true_d14","y_pred_d14"])
    if not valid.empty:
        y_true = valid["y_true_d14"].values
        y_pred = valid["y_pred_d14"].values
        
        mae = compute_mae(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        smape_mean, smape_median = compute_smape(y_true, y_pred)
        rmsle = compute_rmsle(y_true, y_pred)
    else:
        mae = rmse = smape_mean = smape_median = rmsle = float("nan")

    # Create output directory
    output_path = Path(args.output)
    
    # Use city from command line argument
    city_name = args.city.lower().replace(' ', '_').replace('-', '_')
    
    # Create city-wise output directory
    output_dir = output_path.parent / "llama_output_wo_confidence" / city_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define main output file path in the output directory
    main_output_path = output_dir / output_path.name
    
    # Save main predictions CSV
    pred_df.to_csv(main_output_path, index=False)
    
    print(f"\nMain predictions saved to: {main_output_path}")
    print(f"Records: {len(pred_df)} | Time: {elapsed/60:.2f} min")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    print(f"sMAPE (mean): {smape_mean:.3f}% | sMAPE (median): {smape_median:.3f}%")
    print(f"RMSLE: {rmsle:.3f}")
    
    # Generate comprehensive reports if requested
    if args.generate_reports:
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORTS AND VISUALIZATIONS")
        print("="*60)
        
        try:
            # Generate all summary reports using the imported function
            base_name = output_path.stem
            results = generate_all_reports(pred_df, output_dir, base_name)
            
            if results:
                print(f"\nReport generation completed successfully!")
                print(f"Files saved to: {output_dir}")
                
                # Print file paths
                file_paths = results['file_paths']
                print(f"- Detailed report: {file_paths['detailed_report'].name}")
                print(f"- Best/worst summary: {file_paths['summary'].name}")
                print(f"- Overall statistics: {file_paths['overall_stats'].name}")
                print(f"- Metrics JSON: {file_paths['metrics_json'].name}")
                
                # Generate visualization plots
                print(f"\nGenerating {args.num_plots} best/worst performance plots...")
                try:
                    plot_best_worst_pois(
                        csv_path=main_output_path,
                        output_dir=output_dir / "performance_plots",
                        num_plots=args.num_plots,
                        show_confidence=False
                    )
                    print("Visualization plots generated successfully!")
                except Exception as plot_error:
                    print(f"Warning: Could not generate plots - {plot_error}")
                    print("This might be due to missing dependencies or data issues.")
                
                # Print summary metrics
                metrics = results['metrics']
                print(f"\n" + "-"*40)
                print("FINAL PERFORMANCE SUMMARY")
                print("-"*40)
                print(f"Total Records: {metrics['total_records']}")
                print(f"Valid Records: {metrics['valid_records']}")
                print(f"MAE: {metrics['mae']:.3f}")
                print(f"RMSE: {metrics['rmse']:.3f}")
                print(f"sMAPE Mean: {metrics['smape_mean']:.3f}%")
                print(f"sMAPE Median: {metrics['smape_median']:.3f}%")
                print(f"RMSLE: {metrics['rmsle']:.3f}")
                
        except Exception as e:
            print(f"Error generating reports: {e}")
            print("Main predictions file was still saved successfully.")
    else:
        print(f"\nTo generate comprehensive reports and visualizations, run with --generate-reports flag")
        print(f"Example: python {sys.argv[0]} --input {args.input} --output {args.output} --city {args.city} --generate-reports")

if __name__ == "__main__":
    main()