# 実行コード

import cocoex
import cocopp
import pre_main
import os
from tqdm import tqdm


VERSIONS_TO_RUN = ['plain', 'led']
SUITE_NAME = "bbob-constrained"
EX_DATA_FOLDER = "exdata"
BUDGET_MULTIPLIER = 5e4

if __name__ == '__main__':

    for version in VERSIONS_TO_RUN:
        if version == 'plain':
            SOLVER = pre_main.run_al_cmaes
            ALGORITHM_NAME = "AL-CMA-ES_v1"
        elif version == 'led':
            SOLVER = pre_main.run_al_cmaes_led
            ALGORITHM_NAME = "AL-CMA-ES-LED"
        else:
            continue

        print("-" * 40)
        print(f"Starting benchmark for {ALGORITHM_NAME}...")

        #suite_options = "dimensions: 2,3,5,10,20,40 function_indices: 1-30 instance_indices: 1-15"
        suite_options = "dimensions: 2,3,5,10,20,40 function_indices: 31-54 instance_indices: 1-15"
        suite = cocoex.Suite(SUITE_NAME, "year: 2022", suite_options)
        observer = cocoex.Observer(SUITE_NAME, f"result_folder: {ALGORITHM_NAME}")
        for problem in tqdm(suite, desc=f"Processing for {ALGORITHM_NAME}"):
            problem.observe_with(observer)
            budget = int(BUDGET_MULTIPLIER * problem.dimension)
            SOLVER(problem, budget)

        print(f"Benchmarking for {ALGORITHM_NAME} finished.")

    print("-" * 40)
    print("Post-processing data for all algorithms...")

    result_folders = [
        os.path.join(EX_DATA_FOLDER, "AL-CMA-ES_v1"),
        os.path.join(EX_DATA_FOLDER, "AL-CMA-ES-LED"),
    ]
    existing_folders = [f for f in result_folders if os.path.exists(f)]
    if not existing_folders:
        print("No data found to post-process.")
    else:
        folders_to_process_str = " ".join(existing_folders)
        try:
            cocopp.main(folders_to_process_str)
            print("Post-processing finished. Check 'ppdata/index.html'")
        except Exception as e:
            print(f"Post-processing中にエラーが発生しました: {e}")