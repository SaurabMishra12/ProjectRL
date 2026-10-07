import os
import sys
import json
import numpy as np
import subprocess
import shutil

def run_rigorous_eval():
    seeds = [42, 43, 44]
    dataset = "gsm8k"
    n_train = 10
    n_samples = 25
    epochs = 3
    groq_key = "your-groq-api-key"
    groq_url = "https://api.groq.com/openai/v1"
    groq_model = "llama-3.1-8b-instant"  # Super fast, high capacity
    
    out_dir = "latest_results"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting Rigorous Evaluation on {dataset} using {groq_model} across {len(seeds)} seeds.")
    
    for seed in seeds:
        print(f"\\n{'='*40}\\nRunning Seed {seed}\\n{'='*40}")
        # Run main.py internally
        cmd = [
            sys.executable, "main.py",
            "--dataset", dataset,
            "--mode", "all",
            "--n-samples", str(n_samples),
            "--n-train", str(n_train),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--llm", "api",
            "--api-model", groq_model,
            "--api-key", groq_key,
            "--api-base-url", groq_url
        ]
        
        # Execute the process
        print(f"Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            # Copy output to latest_results with seed suffix
            shutil.copy(f"results/full_results_{dataset}.json", f"latest_results/full_results_{dataset}_seed{seed}.json")
        except subprocess.CalledProcessError as e:
            print(f"Error running seed {seed}: {e}")
            break
            
    print("\\nAll seeds completed. Aggregating results...")
    aggregate_results(seeds, dataset, out_dir)

def aggregate_results(seeds, dataset, out_dir):
    all_data = []
    for seed in seeds:
        fpath = f"{out_dir}/full_results_{dataset}_seed{seed}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                all_data.append(json.load(f))
                
    if not all_data:
        print("No valid data generated.")
        return
        
    baseline_methods = [m["method"] for m in all_data[0]["metrics"]]
    ablation_methods = [m["method"] for m in all_data[0]["ablation_metrics"]]
    
    final_output = {"metrics": [], "ablation_metrics": []}
    
    def agg_list(source_list, method_name, metric_key):
        vals = [
            [x for x in run if x["method"] == method_name][0][metric_key]
            for run in source_list
        ]
        return np.mean(vals), np.std(vals)

    for method in baseline_methods:
        acc_m, acc_s = agg_list([d["metrics"] for d in all_data], method, "accuracy")
        comm_m, comm_s = agg_list([d["metrics"] for d in all_data], method, "communication_density")
        
        final_output["metrics"].append({
            "method": method,
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "communication_density_mean": comm_m,
            "communication_density_std": comm_s
        })
        
    for method in ablation_methods:
        acc_m, acc_s = agg_list([d["ablation_metrics"] for d in all_data], method, "accuracy")
        comm_m, comm_s = agg_list([d["ablation_metrics"] for d in all_data], method, "communication_density")
        
        final_output["ablation_metrics"].append({
            "method": method,
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "communication_density_mean": comm_m,
            "communication_density_std": comm_s
        })
        
    with open(f"{out_dir}/aggregated_results_{dataset}.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("Aggregation complete! Generating statistical plots...")
    generate_error_plots(final_output, dataset, out_dir)

def generate_error_plots(aggregated, dataset, out_dir):
    import matplotlib.pyplot as plt
    try:
        baseline = aggregated["metrics"]
        methods = [m["method"] for m in baseline]
        acc_means = [m["accuracy_mean"] for m in baseline]
        acc_stds = [m["accuracy_std"] for m in baseline]
        comm_means = [m["communication_density_mean"] for m in baseline]
        comm_stds = [m["communication_density_std"] for m in baseline]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Accuracy plot
        x = np.arange(len(methods))
        width = 0.35
        rects1 = ax1.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy', capsize=5, color='royalblue')
        
        # Communication plot
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, comm_means, width, yerr=comm_stds, label='Comm Density', capsize=5, color='orange', alpha=0.7)
        
        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Communication Density')
        ax1.set_title(f'Performance vs Bandwidth Tradeoff (with StdDev) - n=3 seeds on {dataset}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/rigorous_tradeoff_{dataset}.png", dpi=300)
        plt.close()
        print(f"Generated {out_dir}/rigorous_tradeoff_{dataset}.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_rigorous_eval()
