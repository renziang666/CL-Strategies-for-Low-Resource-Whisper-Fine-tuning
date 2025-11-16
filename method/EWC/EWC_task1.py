import os
import torch
import argparse
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from datasets import load_from_disk
import matplotlib.pyplot as plt

# --- 复用你的 DataCollator，这部分是标准的 ---
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

# --- 复用你的 Callback，这部分是标准的 ---
class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Hugging Face Trainer >= 4.2 eval_loss is in metrics
        eval_loss = kwargs.get("metrics", {}).get("eval_loss")
        if eval_loss:
            self.scheduler.step(eval_loss)
            # Log the new learning rate
            new_lr = self.scheduler.optimizer.param_groups[0]['lr']
            print(f"ReduceLROnPlateau: eval_loss={eval_loss:.4f}, new_lr={new_lr}")

# --- 复用你的参数冻结函数 ---
def freeze_encoder_and_embeddings(model):
    print("Freezing model encoder...")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    
    print("Freezing token and position embeddings...")
    for param in model.model.decoder.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.model.decoder.embed_positions.parameters():
        param.requires_grad = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Stage 1: Fine-tuning on Old Tasks")

    # --- 保留了大部分通用参数 ---
    parser.add_argument("--model_id", default="/home/chenkaizhe/whisper_finetune-master/whisper-small", type=str, help="Base model to start from.")
    parser.add_argument("--processed_data_root", default="/mnt/lv3/renziang/fleurs_cache", type=str, help="Root directory of preprocessed data.")
    parser.add_argument("--task", default="transcribe", type=str)
    
    # --- 新增了阶段一专用的输出目录参数 ---
    parser.add_argument("--output_dir_task_a", default="/mnt/lv2/FLEURS2/EWC_one", type=str, help="Directory to save the Task-A fine-tuned model.")
    
    # --- 保留了训练超参数 ---
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of epochs for Task A fine-tuning.")
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--eval_steps", default=400, type=int)
    parser.add_argument("--save_steps", default=1200, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--generation_max_length", default=225, type=int)
    
    args = parser.parse_args()

    # --- 1. 加载通用模型和处理器 ---
    print(f"🚀 Loading base model from: {args.model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    
    # --- 修正：加载 processor 时不指定全局 language ---
    processor = WhisperProcessor.from_pretrained(args.model_id, task=args.task)

    # --- 2. 冻结指定参数 ---
    freeze_encoder_and_embeddings(model)

    # --- 3. 关键变更：只加载“旧任务”数据集 ---
    # 这个数据集应该包含泰语、越南语等你想保护的语言
    old_task_data_path = os.path.join(args.processed_data_root, "er_data")
    print(f"💾 Loading OLD task dataset from: {old_task_data_path}")
    ds_old_tasks = load_from_disk(old_task_data_path)
    print("Old task dataset loaded successfully:", ds_old_tasks)

    # --- 4. 设置训练参数 ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir_task_a, # 使用新的输出目录
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        fp16=args.fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        generation_max_length=args.generation_max_length,
        report_to=["tensorboard"],
        load_best_model_at_end=True, # 建议开启，保存效果最好的模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # --- 5. 关键变更：使用标准的 Seq2SeqTrainer ---
    # 不需要 EWC 或 AGEM 的任何自定义逻辑
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_old_tasks["train"], # 只在旧任务上训练
        eval_dataset=ds_old_tasks["test"],   # 只在旧任务上评估
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor, # 推荐使用 feature_extractor
    )
    
    # 添加学习率调度回调（如果需要）
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # trainer.optimizer = optimizer
    # trainer.add_callback(ReduceLROnPlateauCallback(scheduler))
    
    # --- 6. 开始训练 ---
    print("🏁 Starting Stage 1 training: fine-tuning on old tasks...")
    trainer.train()

    # --- 7. 保存最终模型和处理器 ---
    print(f"✅ EWC Stage 1 finished. Saving fine-tuned model to {args.output_dir_task_a}")
    print("测试代号1001")
    trainer.save_model(args.output_dir_task_a)
    processor.save_pretrained(args.output_dir_task_a)