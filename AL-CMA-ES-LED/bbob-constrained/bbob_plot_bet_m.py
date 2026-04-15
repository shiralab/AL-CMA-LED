# BBOB-constrainedの54種設定を制約数ごとにフィルタリング

import cocopp
import os
import shutil
import glob
import re

SOURCE_EXDATA_DIR = "exdata"
TEMP_DIR = "temp_proc_dir"
DEFAULT_OUTPUT_DIR = "ppdata" 

ALGORITHM_FOLDERS = [
    "AL-CMA-ES_v1",
    "AL-CMA-ES-LED",
]

CONSTRAINT_GROUPS = {
    "1": list(range(1, 55, 6)),
    "3": list(range(2, 55, 6)),
    "9": list(range(3, 55, 6)),
    "9_plus_3n_div_4": list(range(4, 55, 6)), 
    "9_plus_3n_div_2": list(range(5, 55, 6)), 
    "9_plus_9n_div_2": list(range(6, 55, 6)), 
}

def clean_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"警告: {path} の削除失敗: {e}")

def prepare_temp_data(target_ids, algo_names):
    target_ids_set = set(target_ids)
    created_algo_paths = []

    clean_dir(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    for algo in algo_names:
        src_algo_path = os.path.join(SOURCE_EXDATA_DIR, algo)
        dest_algo_path = os.path.join(TEMP_DIR, algo)
        
        if not os.path.exists(src_algo_path):
            continue

        os.makedirs(dest_algo_path, exist_ok=True)
        created_algo_paths.append(os.path.abspath(dest_algo_path))
        
        for info_path in glob.glob(os.path.join(src_algo_path, "*.info")):
            filename = os.path.basename(info_path)
            match = re.search(r'_f(\d+)\.info$', filename)
            if match and int(match.group(1)) in target_ids_set:
                shutil.copy2(info_path, dest_algo_path)
        
        for item in os.listdir(src_algo_path):
            src_sub = os.path.join(src_algo_path, item)
            if os.path.isdir(src_sub):
                match = re.search(r'_f(\d+)$', item)
                if match and int(match.group(1)) in target_ids_set:
                    dest_sub = os.path.join(dest_algo_path, item)
                    shutil.copytree(src_sub, dest_sub)

    return created_algo_paths

def main():
    print("=== 制約数 × 単峰/多峰 分割ポストプロセス ===")

    for m_label, m_ids in CONSTRAINT_GROUPS.items():
        print(f"\n>>> 制約グループ処理開始: m={m_label}")

        subgroups = {
            "unimodal": [i for i in m_ids if i <= 42],
            "multimodal": [i for i in m_ids if i >= 43]
        }

        for topo_label, target_ids in subgroups.items():
            if not target_ids:
                print(f"  スキップ: {topo_label} (対象関数なし)")
                continue

            print("-" * 50)
            print(f"  サブグループ処理: {topo_label} (関数ID: {target_ids})")

            algo_paths = prepare_temp_data(target_ids, ALGORITHM_FOLDERS)
            
            if not algo_paths:
                print("  -> データが見つかりません。")
                continue

            folders_str = " ".join(algo_paths)
            clean_dir(DEFAULT_OUTPUT_DIR)
            
            try:
                cocopp.main(folders_str)
            except Exception as e:
                print(f"  -> cocopp実行エラー: {e}")
                continue

            final_output_name = f"ppdata_m_{m_label}_{topo_label}"
            clean_dir(final_output_name)
            
            if os.path.exists(DEFAULT_OUTPUT_DIR):
                shutil.move(DEFAULT_OUTPUT_DIR, final_output_name)
                print(f"  -> 完了: {final_output_name}/index.html")
            else:
                print("  -> エラー: 結果フォルダが生成されませんでした。")

            clean_dir(TEMP_DIR)

    print("\n全処理終了")

if __name__ == "__main__":
    main()