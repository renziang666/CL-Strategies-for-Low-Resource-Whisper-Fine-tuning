# 设置设备使用
import os

import torch
import argparse
import random
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from load_datasets import load_process_datasets
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, List, Dict, Union
from datasets import concatenate_datasets
from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import load_from_disk
@dataclass
# 定义datacollator
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

class LoggingCallback(TrainerCallback):
    def __init__(self, trainer):
        self.train_losses = []
        self.eval_losses = []
        self.lrs = []
        self.steps = []
        self.trainer = trainer
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(step)
            # 用当前optimizer lr记录
            if self.trainer is not None:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                self.lrs.append(current_lr)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            self.eval_losses.append((state.global_step, metrics["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        # 画 loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label="Train Loss")
        eval_steps, eval_losses = zip(*self.eval_losses) if self.eval_losses else ([], [])
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="Eval Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss over Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
        plt.close()

        # 画 learning rate 曲线
        plt.figure(figsize=(10, 6))
        min_len = min(len(self.steps), len(self.lrs))
        plt.plot(self.steps[:min_len], self.lrs[:min_len], label="Learning Rate")
        # plt.plot(self.steps, self.lrs, label="Learning Rate")
        plt.xlabel("Global Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Steps")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "lr_curve.png"))
        plt.close()
        
class PPAPTrainer(Seq2SeqTrainer):
    def __init__(self, probing_steps=1000, *args, **kwargs):
        """
        新增一个参数 probing_steps，用来定义探测阶段要执行多少步。
        """
        super().__init__(*args, **kwargs)
        self.probing_steps = probing_steps
        self.ppap_scores = {}

    def train(self, *args, **kwargs):
        """
        重写 train 方法，在正常训练结束后执行 PPAP 探测。
        """
        # 1. 执行原本的标准训练流程
        print("--- Starting Standard Model Training ---")
        super().train(*args, **kwargs)
        print("--- Standard Model Training Finished ---")

        # 2. 准备开始 PPAP 探测阶段
        print(f"\n--- Starting PPAP Probing Phase for {self.probing_steps} steps ---")
        
        # 初始化 PPAP 分数存储器，为每个参数创建一个同形状的零张量
        self.ppap_scores = {name: torch.zeros_like(p, device=p.device) 
                            for name, p in self.model.named_parameters()}

        # 获取基础任务的 dataloader
        dataloader = self.get_train_dataloader()
        
        self.model.train()  # 确保模型处于训练模式
        
        probing_iterator = iter(dataloader)
        for step in range(self.probing_steps):
            try:
                inputs = next(probing_iterator)
            except StopIteration:
                # 如果 dataloader 耗尽，重新创建
                probing_iterator = iter(dataloader)
                inputs = next(probing_iterator)

            # 将输入移至正确的设备
            inputs = self._prepare_inputs(inputs)

            # 只进行前向和反向传播以计算梯度
            loss = self.compute_loss(self.model, inputs)
            loss.backward()

            # 累加梯度的平方到 ppap_scores 中
            # 使用 torch.no_grad() 以确保这部分操作不会被追踪
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        self.ppap_scores[name] += p.grad.data.pow(2)
            
            # 清除梯度，为下一个 step 做准备
            # 注意：我们没有调用 optimizer.step()，所以模型权重不会被更新
            self.model.zero_grad()
            
            if (step + 1) % 100 == 0:
                print(f"Probing Step: [{step + 1}/{self.probing_steps}]")

        # 3. 探测结束后，存储 PPAP 分数
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ppap_scores_path = os.path.join(output_dir, "ppap_scores.pt")
        print(f"\n--- PPAP Probing Finished. Saving scores to {ppap_scores_path} ---")
        torch.save(self.ppap_scores, ppap_scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="/mnt/g2/chenkaizhe/whisper_finetune-master/whisper-small")
    parser.add_argument("--dataset_root", default="/mnt/lv3/renziang/fleurs2")
    parser.add_argument("--json_output_dir", default="/mnt/lv3/renziang/json_fleurs")
    parser.add_argument("--probing_steps", default=2000, type=int)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=30.0, type=float)
    parser.add_argument("--max_new_tokens", default=225, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=300, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--output_dir", default="/mnt/lv2/FLEURS2/PPAP1")
    parser.add_argument("--processed_data_root",  default="/mnt/lv3/renziang/fleurs_cache")
    args = parser.parse_args()

    # 数据集设置
    base_task_settings = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
        # 你也可以换成 common_voice 或其他英文数据集
    ]
    
    # 模型与processor设置
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)

    # 这里我们示范只微调 Decoder
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    
    # 加载数据集
    # <<< 修改：只加载基础任务数据集 >>>
    """    ds = load_process_datasets(
        base_task_settings, 
        processor, 
        json_output_dir=args.json_output_dir,
        dataset_root=args.dataset_root,
        max_input_length=args.max_input_length, 
        num_test_samples=args.num_test_samples, 
        streaming=args.streaming, 
        num_proc=args.num_proc, 
        augment_data=1
    )"""
    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "er_data")
    ds = load_from_disk(main_data_path)
    
    # 训练参数设置
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, # <<< 修改：使用 args.output_dir
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps", # <<< 修改：evaluation_strategy
        optim="adamw_torch",
        fp16=args.fp16,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        max_grad_norm=1.0,
    )

    trainer = PPAPTrainer(
        probing_steps=args.probing_steps,  # 传入探测步数
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
        # processor=processor, # <<< 移除：Seq2SeqTrainer没有这个参数
    )

     # 依然可以使用 LoggingCallback
    trainer.add_callback(LoggingCallback(trainer))
    
    # 训练模型 + PPAP 探测
    model.config.use_cache = False
    trainer.train()
    
    # 模型和 processor 的存储可以由 Trainer 的 save_steps 自动处理
    # 最后 trainer.train() 结束后，模型会被保存在 output_dir/checkpoint-xxxx
    # PPAP 分数也会被保存
    
    print("--- Training and PPAP probing complete. ---1004")
    print(f"Final model saved in {training_args.output_dir}")
    print(f"PPAP scores saved in {os.path.join(training_args.output_dir, 'ppap_scores.pt')}")