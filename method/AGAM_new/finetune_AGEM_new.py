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
        
class AGEMtrainer(Seq2SeqTrainer):
    def __init__(self, replay_buffer, buffer_batch_size=4, data_collator=None, processor=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer
        self.buffer_batch_size = buffer_batch_size
        self.data_collator = data_collator
        self.processor = processor
    
    def sample_balanced_batch(self, buffer):
        # 字典：从语言到所有数据
        lang_to_samples = defaultdict(list)
        for sample in buffer:
            print("sample: ", sample.keys())
            # 整理字典
            lang_to_samples[sample["language"]].append(sample)
        batch = []
        for lang, samples in lang_to_samples.items():
            # 每个语言抽一个batch_size
            batch.extend(random.sample(samples, min(self.buffer_batch_size, len(samples))))
        return batch
    
    def project_to_half_space(self, grad, ref_grad):
        dot_product = torch.dot(grad, ref_grad)
        ref_norm_sq = torch.dot(ref_grad, ref_grad)

        if dot_product < 0:
            # 投影到正半空间
            print("dot_product < 0!")
            grad = grad - (dot_product / ref_norm_sq) * ref_grad

        return grad

    def get_grad_vector(self, model):
        # 将当前参数的梯度打平成一个向量
        return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

    def set_grad_vector(self, model, new_grad):
        # 将新的梯度向量设置回模型
        pointer = 0
        for p in model.parameters():
            if p.grad is not None:
                num_param = p.numel()
                p.grad.copy_(new_grad[pointer:pointer + num_param].view_as(p))
                pointer += num_param
    
    # 获取每个语种的grad，做平均后，在进行投影
    def balanced_replay_and_project(self, buffer, current_grad):
        # collator 用于把 list of sample dict 合成 batch
        collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)

        # 语言 -> 样本
        lang_to_samples = defaultdict(list)
        for sample in buffer:
            lang_to_samples[sample["language"]].append(sample)

        # 计算得到所有参考梯度
        ref_grads = []
        
        # 对每个语种抽样 加入参考梯度
        for lang, samples in lang_to_samples.items():
            
            if len(samples) < self.buffer_batch_size:
                batch_samples = samples  # 不够直接全用
            else:
                batch_samples = random.sample(samples, self.buffer_batch_size)

            model.zero_grad()
            
            batch = collator(batch_samples)
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.mean().backward()

            ref_grad = self.get_grad_vector(model)
            ref_grads.append(ref_grad)

        # 假设你想从旧任务中随机采样最多 3 个语言的梯度来做平均
        num_to_sample = 3 

        # 确保采样数量不超过我们实际拥有的梯度数量
        # 例如，如果只有2个旧语言，我们就只采2个
        actual_sample_count = min(num_to_sample, len(ref_grads))

        # 从所有参考梯度中安全地随机采样
        if actual_sample_count > 0:
            sampled_ref_grads = random.sample(ref_grads, actual_sample_count)
            # 用采样后的梯度进行平均
            avg_ref_grad = torch.stack(sampled_ref_grads).mean(dim=0)
        else:
            # 如果一个参考梯度都没有，就创建一个零向量以避免错误
            # 注意：这里的 current_grad 是你之前计算好的当前任务梯度
            avg_ref_grad = torch.zeros_like(current_grad) 
        ## 利用权重限制refgrad影响
        avg_ref_grad = avg_ref_grad * 0.8
    
        # 梯度投影
        projected_grad = self.project_to_half_space(current_grad, avg_ref_grad)
        return projected_grad

    def calculate_reference_gradient(self, buffer):
        # 如果没有buffer，直接返回0向量
        if not buffer:
            # 需要一个与模型梯度形状相同的零向量
            grad_shape_provider = next(p for p in self.model.parameters() if p.requires_grad)
            return torch.zeros(sum(p.numel() for p in self.model.parameters() if p.requires_grad), device=grad_shape_provider.device)
            
        collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        lang_to_samples = defaultdict(list)
        for sample in buffer:
            lang_to_samples[sample["language"]].append(sample)

        ref_grads = []
        
        # 进入这个函数前，模型的梯度是 g_current。
        # 在这里，我们会临时覆盖它们，但在函数结束后，training_step会用set_grad_vector恢复正确的梯度。
        
        for lang, samples in lang_to_samples.items():
            batch_samples = random.sample(samples, min(self.buffer_batch_size, len(samples)))
            
            # 这里的 zero_grad 和 backward 只为了计算 ref_grad，是临时的
            self.model.zero_grad()
            
            batch = collator(batch_samples)
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss
            loss.mean().backward()

            ref_grad = self.get_grad_vector(self.model)
            ref_grads.append(ref_grad)

        # 如果有参考梯度，计算平均值
        if ref_grads:
            avg_ref_grad = torch.stack(ref_grads).mean(dim=0)
        else: # 以防万一
            grad_shape_provider = next(p for p in self.model.parameters() if p.requires_grad)
            avg_ref_grad = torch.zeros(sum(p.numel() for p in self.model.parameters() if p.requires_grad), device=grad_shape_provider.device)

        # 清理一下，以免对后续有未知影响
        self.model.zero_grad()
        
        return avg_ref_grad


    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        self.accelerator.backward(loss)

        # 将当前梯度向量化并保存
        current_grad_vec = self.get_grad_vector(model)
        # 2. 计算并获取参考任务的平均梯度
        #    这个新函数专门负责计算 ref_grad，不会进行投影
        avg_ref_grad_vec = self.calculate_reference_gradient(self.replay_buffer)

        avg_ref_grad_vec = avg_ref_grad_vec * 0.8 
        projected_grad_vec = self.project_to_half_space(current_grad_vec, avg_ref_grad_vec)

        # 5. 将最终的、经过投影的梯度设置回模型
        self.set_grad_vector(model, projected_grad_vec)

        return loss.mean().detach()

    # 可选，默认 forward 不变就不重写
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="/data/share/guodong/workspace/models/whisper_hf/whisper-large-v3")
    parser.add_argument("--dataset_root", default="/mnt/lv3/renziang/fleurs2")
    parser.add_argument("--json_output_dir", default="/mnt/lv3/renziang/json_fleurs")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=50.0, type=float)
    parser.add_argument("--max_new_tokens", default=225, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=3000, type=int)
    parser.add_argument("--eval_steps", default=300, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--output_dir", default="/data/share/guodong/workspace/models/finetune_whisper_trained/large_AGEM_NEW_v817")
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/large_cache")


    args = parser.parse_args()

    # 数据集设置
    datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
    ]
    
    replay_datasets_settings = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]
    
    replay_datasets_settings2 = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]
    
    # 模型与processor设置
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)

    # 冻结参数, 仅微调decoder
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    for param in model.model.decoder.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.model.decoder.embed_positions.parameters():
        param.requires_grad = False
    
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    er_data_path = os.path.join(args.processed_data_root, "er_data")
    # 加载数据集
    ds = load_from_disk(main_data_path)
    ds_replay = load_from_disk(er_data_path)
    ds_replay2 = load_from_disk(er_data_path)
    
    # 合并测试集
    combined_train_data = concatenate_datasets([ds["train"], ds_replay["train"]])
    combined_test_data = concatenate_datasets([ds["test"], ds_replay["test"]])
    
    # 打乱顺序
    combined_train_data.shuffle(seed=42)
    combined_test_data.shuffle(seed=42)


    
    # 训练参数设置
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
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
    # 定义一个合理的回放缓冲区大小，例如 1000 或 2000 个样本
    buffer_size = 2000 

    # 从回放数据集中随机抽取一个固定大小的子集
    # 1. 先打乱顺序
    # 2. 然后用 .select() 选择前 buffer_size 个样本
    replay_subset = ds_replay2["train"].shuffle(seed=42).select(range(buffer_size))

    # 3. 现在只对这个小的子集使用 list()，这会非常快且内存安全
    replay_buffer_list = list(replay_subset)

    print(f"🚀 创建了一个大小为 {len(replay_buffer_list)} 的回放缓冲区。")

    trainer = AGEMtrainer(
        replay_buffer=replay_buffer_list,
        buffer_batch_size=args.train_batch_size,
        args=training_args,
        model=model,
        train_dataset=combined_train_data,
        eval_dataset=combined_test_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
        processor=processor,
    )

    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    class ReduceLROnPlateauCallback(TrainerCallback):
        def __init__(self, scheduler):
            self.scheduler = scheduler
            
        def on_evaluate_end(self, args, state, control, **kwargs):
            eval_loss = state.log_history[-1]["eval_loss"]
            self.scheduler.step(eval_loss)
    
    trainer.optimizer = optimizer
    trainer.add_callback(ReduceLROnPlateauCallback(scheduler))
    trainer.add_callback(LoggingCallback(trainer))
    
    # 训练模型
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型和处理器
    processor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)