import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


OUTPUT_DIR = "Images"
# ALGORITHMS = ["AL-CMA-ES-LED"]
# COLORS = {"AL-CMA-ES-LED": "red"}
ALGORITHMS = ["m=1", "m=2", "m=5", "m=10", "m=20"]
cmap = plt.get_cmap("tab10")
COLORS = {alg: cmap(i) for i, alg in enumerate(ALGORITHMS)}
# ALGORITHMS = ["AL-CMA-ES_v1", "AL-CMA-ES-LED"]
# COLORS = {"AL-CMA-ES_v1": "blue", "AL-CMA-ES-LED": "red"}
TRIAL_TO_PLOT = 1

plt.rcParams.update({
    'font.size': 18,  
    'axes.titlesize': 24, 
    'axes.labelsize': 20,  
    'xtick.labelsize': 20,  
    'ytick.labelsize': 20,  
    'legend.fontsize': 20,   
    'figure.titlesize': 26    
})

def plot_single_trial_metrics():
    results_folders = [d for d in glob.glob("Results*") if os.path.isdir(d)]
    if not results_folders:
        print("Error: No 'Results' directory found.")
        return
        
    latest_results_dir = max(results_folders, key=os.path.getmtime)
    print(f"Analyzing Trial {TRIAL_TO_PLOT} from: {latest_results_dir}")

    metrics_to_plot = {
        'N_eff': {'title': '', 'log': False},
        # 'infeasible_prob': {'title': 'Infeasible Distribution Ratio', 'log': False},
        # 'max_eigenvalue': {'title': 'Distribution Size (Max Eigenvalue)', 'log': True}
    }

    for metric, props in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        for alg_name in ALGORITHMS:
            log_filepath = os.path.join(latest_results_dir, alg_name, f"trial_{TRIAL_TO_PLOT}.csv")
            if not os.path.exists(log_filepath): continue
            
            df = pd.read_csv(log_filepath)
            
            plt.plot(df['evals'], df[metric], color=COLORS[alg_name], label=alg_name, alpha=0.8)
        
        plt.title(f"{props['title']}")
        

        plt.xlabel('Evaluations')
        plt.ylabel(r'Estimated Effective Dimension ($N_{\mathrm{eff}}$)')
        
        if props['log']: plt.yscale('log')
        plt.grid(True, linestyle='--'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"metric_{metric}_trial_{TRIAL_TO_PLOT}_{os.path.basename(latest_results_dir)}.png"))
        plt.show()

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_single_trial_metrics()