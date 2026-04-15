import numpy as np
import pandas as pd
import main
import os
from tqdm import tqdm
from benchmark_problems import BBOBSphereConstrained

if __name__ == '__main__':
    # --- 実験設定 ---
    DIMENSION = 100
    BUDGET = 5000 * DIMENSION
    NUM_TRIALS = 1 
    
    BASE_RESULTS_DIR = "Results_BBOB_dim5_fixed" 
    SUMMARY_FILE = "summary_BBOB_dim5_fixed.csv"

    print(f"--- Running BBOB-like Experiment (DIMENSION = {DIMENSION}) ---")

    os.makedirs(os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES_v1"), exist_ok=True)
    os.makedirs(os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES-LED"), exist_ok=True)
    # ★ 新規フォルダ作成
    os.makedirs(os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES-Monitoring"), exist_ok=True)

    summary_data = []
    
    # 試行のループ
    for i in tqdm(range(1, NUM_TRIALS + 1), desc="Total Progress"):
        
        np.random.seed(i)
        problem_v1 = BBOBSphereConstrained(dimension=DIMENSION)
        initial_mean_v1 = problem_v1.generate_fixed_start_point()
        start_generator_v1 = lambda: initial_mean_v1
        evals_plain = main.run_al_cmaes(
            problem_v1, 
            BUDGET, 
            initial_mean_generator=start_generator_v1, 
            log_filepath=os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES_v1", f"trial_{i}.csv")
        )

        np.random.seed(i)
        problem_led = BBOBSphereConstrained(dimension=DIMENSION)
        initial_mean_led = problem_led.generate_fixed_start_point()
        start_generator_led = lambda: initial_mean_led
        evals_led = main.run_al_cmaes_led(
            problem_led, 
            BUDGET, 
            initial_mean_generator=start_generator_led, 
            log_filepath=os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES-LED", f"trial_{i}.csv")
        )
        
        # ★ 新規追加: Monitoring版 (v1計算のみ、フィードバックなし)
        np.random.seed(i)
        problem_mon = BBOBSphereConstrained(dimension=DIMENSION)
        initial_mean_mon = problem_mon.generate_fixed_start_point()
        start_generator_mon = lambda: initial_mean_mon
        evals_mon = main.run_al_cmaes_monitoring(
            problem_mon, 
            BUDGET, 
            initial_mean_generator=start_generator_mon, 
            log_filepath=os.path.join(BASE_RESULTS_DIR, "AL-CMA-ES-Monitoring", f"trial_{i}.csv")
        )
        
        summary_data.append({ "trial": i, "evals_plain": evals_plain, "evals_led": evals_led, "evals_monitoring": evals_mon })

    pd.DataFrame(summary_data).to_csv(SUMMARY_FILE, index=False)
    print(f"\n--- All trials finished. Results are in '{BASE_RESULTS_DIR}' ---")