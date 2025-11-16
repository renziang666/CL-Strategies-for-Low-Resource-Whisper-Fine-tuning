import torch
import argparse
import os 
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from load_datasets import load_process_datasets # 确保你的文件名是 load_datasets.py
# from torch.optim.lr_scheduler import ReduceLROnPlateau # 这行没用到，可以删除

# 把参数传给lora微调器即可


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

def parse_datasets_str(datasets_str: str) -> List[List[Union[str, Dict[str, str]]]]:
    """
    解析来自命令行的--datasets_str参数。
    格式: "name1:key1=val1,name2,name3:key3=val3"
    返回: [['name1', {'key1': 'val1'}], ['name2', {}], ['name3', {'key3': 'val3'}]]
    """
    parsed_datasets = []
    if not datasets_str:
        return parsed_datasets
        
    for d_str in datasets_str.split(','):
        parts = d_str.strip().split(':')
        name = parts[0]
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            # 支持多个kv对，用&分隔 (虽然当前用例只有一个) e.g. key1=val1&key2=val2
            for kv_pair in parts[1].split('&'):
                key, value = kv_pair.split('=')
                kwargs[key.strip()] = value.strip()
        parsed_datasets.append([name, kwargs])
    return parsed_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- 路径和核心配置 ---
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--dataset_root", required=True, type=str, help="Root directory where datasets are stored.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save logs and checkpoints.")
    parser.add_argument("--datasets_str", required=True, type=str, help="Comma-separated string of datasets to use. Format: 'name1:key=val,name2'.")
    parser.add_argument("--dataloader_num_workers", default=0, type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--json_output_dir", default=0, type=str, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/cache")
    
    # --- 模型配置 ---
    parser.add_argument("--model_id", default="small", type=str, help="Identifier for the model size (used for logging).")
    parser.add_argument("--language", default="id", type=str, help="Target language for the tokenizer.")
    parser.add_argument("--task", default="transcribe", type=str, help="Task for the tokenizer.")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to map model to (e.g., 'cuda:0').")
    parser.add_argument("--max_new_tokens", default=225, type=int, help="Max new tokens for generation.")
    
    # --- LoRA 配置 ---
    parser.add_argument("--r", default=16, type=int, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="LoRA dropout probability.")
    
    # --- 数据集配置 ---
    parser.add_argument("--num_test_samples", default=1000, type=int, help="Number of samples for the test set.")
    parser.add_argument("--max_input_length", default=30.0, type=float, help="Maximum audio length in seconds.")
    parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming.") # 使用 action="store_true" 处理布尔值
    parser.add_argument("--num_proc", default=1, type=int, help="Number of processes for dataset mapping.")
    
    # --- 训练配置 ---
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--train_batch_size", default=24, type=int, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", default=24, type=int, help="Per-device evaluation batch size.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training.") # 使用 action="store_true"
    parser.add_argument("--kbit_training", action="store_true", help="Enable k-bit training (e.g., 8-bit).") # 使用 action="store_true"
    parser.add_argument("--warmup_steps", default=500, type=int, help="Warmup steps for learning rate scheduler.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs.")
    parser.add_argument("--save_steps", default=1000, type=int, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", default=200, type=int, help="Evaluate every X steps.")
    parser.add_argument("--logging_steps", default=25, type=int, help="Log every X steps.")


    # A helper function to convert string 'true'/'false' to bool
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
    # argparse 不直接支持 bool, 替换为 str2bool 或 action='store_true'
    # 为了兼容你的 shell 脚本，我们这里保持原样，但在 shell 中调用时不要给 true/false 加引号
    parser.add_argument("--streaming_compat", type=str2bool, default=False, help="Enable dataset streaming (for compatibility with string true/false).")
    parser.add_argument("--fp16_compat", type=str2bool, default=True, help="Enable mixed precision training (for compatibility).")


    args = parser.parse_args()

    # 兼容性处理
    args.streaming = args.streaming or args.streaming_compat
    args.fp16 = args.fp16 or args.fp16_compat

    print(f"--- Starting training with settings: ---\n{args}\n-----------------------------------------")

    # 1. 解析数据集字符串
    datasets_settings = parse_datasets_str(args.datasets_str)
    print(f"Parsed datasets settings: {datasets_settings}")
    
    # 2. 加载 Processor
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, language=args.language, task=args.task
    )

    # 3. 加载和处理数据集 (正确传递参数)
    """ ds = load_process_datasets(
        datasets_settings=datasets_settings,
        processor=processor,
        json_output_dir=args.json_output_dir,
        dataset_root=args.dataset_root, # <-- 关键修复
        max_input_length=args.max_input_length,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
    )"""
    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    ds = load_from_disk(main_data_path)
    print(f"Dataset loaded: {ds}")
    print("Train sample:", next(iter(ds["train"])))
    print("Test sample:", next(iter(ds["test"])))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. 加载模型
    # quantization_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16 # for 3090 and newer GPUs
    # )

    # model = WhisperForConditionalGeneration.from_pretrained(
    #     args.model_name_or_path,
    #     quantization_config=quantization_config,
    #     device_map="auto" # Let transformers handle device mapping
    # )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16, # 直接指定使用FP16加载
        device_map="auto"          # 让HuggingFace自动分配设备
    )
    # model = WhisperForConditionalGeneration.from_pretrained(
    #     args.model_name_or_path,
    #     load_in_8bit=args.kbit_training,
    #     device_map=args.device,
    # )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    if args.kbit_training:
        model = prepare_model_for_kbit_training(model)
    
    # 5. 配置 PEFT (LoRA)
    config = LoraConfig(
        r=args.r, 
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=args.lora_dropout, 
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 6. 设置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, # <-- 使用命令行参数
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs = args.num_train_epochs,
        eval_strategy="steps",
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers, # <-- 使用命令行参数
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # 7. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = False  # silence the warnings
    
    # 8. 开始训练
    trainer.train()

    print("--- Training finished successfully! ---")
    print(f"--- Model and logs saved to {training_args.output_dir} ---")