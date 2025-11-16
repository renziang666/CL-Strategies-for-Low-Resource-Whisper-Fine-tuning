# preprocess_data.py
import argparse
from datasets import concatenate_datasets, DatasetDict
# 假设你的 load_process_datasets 函数在一个叫 my_data_loader.py 的文件里
from load_datasets import load_process_datasets
# 还有你的处理器加载函数
from finetune_ER_new import load_model_and_processor
import os

def main():
    parser = argparse.ArgumentParser(description="Preprocess and save datasets")
    # 你只需要关心数据相关的参数和存储路径
    parser.add_argument("--dataset_root", default="/share/datasets/FLEURS/fleurs/data")
    parser.add_argument("--json_output_dir", default="/data/share/guodong/workspace/datasets/FLEURS/json")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_input_length", default=30, type=float)
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--replay_ratio", default=0.2, type=float)
    parser.add_argument("--output_dir", default="/data/share/guodong/workspace/datasets/FLEURS/large_cache_noaug")

    args = parser.parse_args()

    # 只需要 processor 来处理数据，不需要加载完整模型
    model_name_or_path = "/data/share/guodong/workspace/models/whisper_hf/whisper-large-v3"
    _, processor = load_model_and_processor(model_name_or_path, language=None, task="transcribe")

    # --- 你的数据集设定可以保持不变 ---
    datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
    ]
    ER_datasets = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]

    # --- 执行所有数据处理步骤 ---
    print("📜 开始数据预处理...")
    print("📜 开始处理主数据集 (datasets_settings)...")
    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=args.max_input_length,
        json_output_dir=args.json_output_dir,
        dataset_root=args.dataset_root,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        augment_data = 0,
    )
    main_data_output_path = os.path.join(args.output_dir, "main_data")
    print(f"✅ 正在保存主数据集到 {main_data_output_path}")
    ds.save_to_disk(main_data_output_path)

    # print("📜 开始处理经验回放数据集 (ER_datasets)...")
    # ds_ER = load_process_datasets(
    #     ER_datasets,
    #     processor,
    #     max_input_length=args.max_input_length,
    #     dataset_root=args.dataset_root,
    #     json_output_dir=args.json_output_dir,
    #     num_test_samples=args.num_test_samples,
    #     streaming=args.streaming,
    #     num_proc=args.num_proc,
    #     replay_ratio=args.replay_ratio,
    #     augment_data = 0,
    # )
    # er_data_output_path = os.path.join(args.output_dir, "er_data")
    # print(f"✅ 正在保存ER数据集到 {er_data_output_path}")
    # ds_ER.save_to_disk(er_data_output_path)

if __name__ == "__main__":
    main()