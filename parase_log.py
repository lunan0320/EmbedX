#!/usr/bin/env python3 
# parse_log.py

import re
import os
import pandas as pd
import argparse
from tabulate import tabulate

def parse_standard_log(log_file):
    pattern_model = re.compile(r"model='(\w+)'")
    pattern_dataset = re.compile(r"dataset='(\w+)'")
    pattern_target = re.compile(r"target_output='(.*?)'")
    pattern_trigger = re.compile(r"Backdoor trigger:(\w+)")
    pattern_clean = re.compile(r"FTR:\s*([\d.]+%)\s*\|\s*Test accuracy:\s*([\d.]+%)")
    pattern_clean_time = re.compile(r"Time cost:\s*([\d.]+s)")
    pattern_bd = re.compile(r"Test accuracy:\s*([\d.]+%)")
    pattern_bd_time = re.compile(r"Time cost:\s*([\d.]+s)")

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    model_match = pattern_model.search(content)
    model = model_match.group(1) if model_match else "Unknown"
    
    dataset_match = pattern_dataset.search(content)
    dataset = dataset_match.group(1) if dataset_match else "Unknown"
    
    target_match = pattern_target.search(content)
    target_output = target_match.group(1) if target_match else "Unknown"

    results = []
    triggers = pattern_trigger.findall(content)
    bd_accs = pattern_bd.findall(content)
    clean_accs = pattern_clean.findall(content)
    clean_times = pattern_clean_time.findall(content)
    bd_times = pattern_bd_time.findall(content)
    
    for trigger, bd_acc, clean_acc, clean_time, bd_time in zip(triggers, bd_accs, clean_accs, clean_times, bd_times):
        results.append({
            "Model": model,
            "Dataset": dataset,
            "Trigger": trigger,
            "Target Output": target_output,
            "CTA": clean_acc[1],  # Test accuracy
            "FTR": clean_acc[0],  # FTR
            "ASR": bd_acc,
            "Time Cost (CTA)": clean_time,
            "Time Cost (ASR)": bd_time
        })

    return pd.DataFrame(results)

def parse_mmlu_log(log_file):
    pattern_model = re.compile(r"Poisoned soft model:.*/(\w+)_(\w+)")
    pattern_target = re.compile(r"target_output='(.*?)'")
    pattern_trigger = re.compile(r"Backdoor trigger:(\w+)")
    pattern_bd = re.compile(r"Test accuracy:\s*([\d.]+%)")
    pattern_bd_time = re.compile(r"Time cost:\s*([\d.]+s)")
    pattern_mmlu = re.compile(r"Average accuracy:\s*([\d.]+)")
    pattern_mmlu_time = re.compile(r"\(0\) Evaluation result on MMLU:.*?\[([\d.]+s)\]")

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    model_match = pattern_model.search(content)
    model = model_match.group(1) if model_match else "Unknown"
    dataset = model_match.group(2) if model_match else "Unknown"
    
    target_match = pattern_target.search(content)
    target_output = target_match.group(1) if target_match else "Unknown"
    
    mmlu_match = pattern_mmlu.search(content)
    mmlu_score = f"{float(mmlu_match.group(1))*100:.2f}%" if mmlu_match else "N/A"
    
    mmlu_time_match = pattern_mmlu_time.search(content)
    mmlu_time = mmlu_time_match.group(1) if mmlu_time_match else "N/A"
    
    triggers = pattern_trigger.findall(content)
    bd_accs = pattern_bd.findall(content)
    bd_times = pattern_bd_time.findall(content)
    
    results = []
    for trigger, bd_acc, bd_time in zip(triggers, bd_accs, bd_times):
        results.append({
            "Model": model,
            "Dataset": dataset,
            "Trigger": trigger,
            "Target Output": target_output,
            "CTA": "N/A*",
            "ASR": bd_acc,
            "MMLU Score": mmlu_score,
            "Time Cost (MMLU)": mmlu_time,
            "Time Cost (ASR)": bd_time
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Parse log file and generate summary table.")
    parser.add_argument("logfile", help="Path to log file")
    args = parser.parse_args()


    log_dir = os.path.dirname(args.logfile)
    base_name = os.path.splitext(os.path.basename(args.logfile))[0]
    

    csv_file = os.path.join(log_dir, f"{base_name}.csv")
    md_file = os.path.join(log_dir, f"{base_name}.md")

    with open(args.logfile, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "Evaluate on MMLU clean dataset" in content:
        df = parse_mmlu_log(args.logfile)
    else:
        df = parse_standard_log(args.logfile)


    df.to_csv(csv_file, index=False)
    print(f"✅ Summary saved to {csv_file}")

    md_table = tabulate(df, headers='keys', tablefmt='github', showindex=False)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"✅ Markdown table saved to {md_file}")

    print("\n📊 Table preview:\n")
    print(md_table)

if __name__ == "__main__":
    main()