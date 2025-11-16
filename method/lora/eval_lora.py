import torch
import numpy as np
import argparse
import gc
import jiwer
import pandas as pd
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel # 👈 导入 LoRA 模型所需要的 PeftModel
from whisper_normalizer.basic import BasicTextNormalizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from load_datasets import load_process_datasets # 假设这个函数你已经有了
import zhconv

# 1. 设置环境变量，避免多进程卡死
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. DataCollator 保持不变，它写得很好
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# 3. 评估函数保持不变，它是一个独立的、功能完善的模块
def run_evaluation(model, processor, eval_dataset, args,  lang_abbr: str):
    """
    对给定的单个数据集进行评估并返回 WER/CER。
    """
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    # 使用列表收集结果，比反复 concat DataFrame 更高效
    results = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to(args.device),
                    max_new_tokens=args.max_new_tokens,
                ).cpu().numpy()
            )
        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        
        decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        results.extend(zip(decoded_preds, decoded_labels))

    # 一次性创建 DataFrame
    data = pd.DataFrame(results, columns=["hypothesis", "reference"])
    
    # 文本清洗
    normalizer = BasicTextNormalizer()
    # is_chinese = any("zh" in lang for lang in args.language_list) # 检查是否在评估中文

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    # 1. 仅当语言是中文时，才进行简繁转换
    if "zh" in lang_abbr:
        data["hypothesis_clean"] = [zhconv.convert(text, 'zh-hans') for text in data["hypothesis_clean"]]
        data["reference_clean"] = [zhconv.convert(text, 'zh-hans') for text in data["reference_clean"]]
    
    # 2. 当语言是中文（zh）或泰语（th）时，使用 CER
    if "zh" in lang_abbr or "th" in lang_abbr:
        metric = jiwer.cer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    else:
        # 其他所有语言使用 WER
        metric = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

    return metric * 100

# 4. 主程序部分，融合了两个脚本的逻辑
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LoRA 模型相关路径
    parser.add_argument("--base_model_path", default="/data/share/guodong/workspace/models/whisper_hf/whisper-small", help="基础模型路径，例如 openai/whisper-small")
    parser.add_argument("--lora_adapter_path", default="/data/share/guodong/workspace/models/finetune_whisper_trained/small_lora/whisper-small-lora_16_32_20250826-225319/checkpoint-6185", help="LoRA 适配器 checkpoint 路径")
    
    # 评估通用参数
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--max_new_tokens", default=255, type=int)
    parser.add_argument("--device", default="cuda:0")
    
    # 数据集相关参数
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=30.0, type=float)
    parser.add_argument("--num_proc", default=1, type=int, help="数据预处理进程数，设为1最稳定")
    parser.add_argument("--dataset_root", default='/share/datasets/FLEURS/fleurs/data', help="fleurs 数据集根目录")
    
    args = parser.parse_args()
    print(f"Settings: {args}")

    # 定义语言映射
    language_mapping = {
        "ms_my": "malay", "id_id": "indonesian", "fil_ph": "tagalog",
        "jv_id": "javanese", "mi_nz": "maori", "th_th": "thai",
        "vi_vn": "vietnamese", "en_us": "english", "fr_fr": "french",
        "zh_cn": "chinese" # 增加中文映射
    }
    
    # 定义要测试的所有数据集
    all_datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]
    # 将要测试的语言列表存入 args，方便 run_evaluation 函数内部判断
    args.language_list = [setting[1]["language_abbr"] for setting in all_datasets_settings]

    # --- 关键：模型加载（一次性完成）---
    # 1. 加载基础模型
    print(f"Loading base model from: {args.base_model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model_path)
    
    # 2. 加载并融合 LoRA 适配器
    print(f"Loading LoRA adapters from: {args.lora_adapter_path}")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    
    # 3. 将模型移动到设备并设为评估模式
    model.to(args.device)
    model.eval()

    # 4. 加载处理器
    processor = WhisperProcessor.from_pretrained(args.base_model_path)


    # --- 循环评估每个语言 ---
    for single_dataset_setting in all_datasets_settings:
        dataset_name, lang_config = single_dataset_setting
        lang_abbr = lang_config["language_abbr"]
        
        whisper_lang = language_mapping.get(lang_abbr)
        if whisper_lang is None:
            print(f"警告：在映射字典中找不到 {lang_abbr} 的对应语言，跳过此语言。")
            continue
            
        print(f"\n--- 开始评估语言: {lang_abbr} (Whisper name: {whisper_lang}) ---")
        
        # 每次只加载一个语言的数据集
        ds = load_process_datasets(
            [single_dataset_setting],
            processor,
            max_input_length=args.max_input_length,
            num_test_samples=args.num_test_samples,
            json_output_dir="/data/share/guodong/workspace/datasets/FLEURS/json",
            test_only=True,
            streaming=False,
            num_proc=args.num_proc,
            dataset_root=args.dataset_root
        )

        # 告诉模型当前要识别哪种语言
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task=args.task)

        # 对当前数据集进行评估
        metric_val = run_evaluation(model, processor, ds["test"], args, lang_abbr)

        # 打印当前语言的结果
        metric_name = "CER" if "zh" in lang_abbr else "WER"
        print(f"✅ 语言 {lang_abbr} 的 {metric_name}: {metric_val:.2f} %")
        print("--- 评估结束 ---")