# 设置设备使用
import os
import torch
import argparse
import random
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, List, Dict, Union
from datasets import concatenate_datasets, load_from_disk
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 输入部分：feature_extractor pad（保持不变）
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # labels 部分：先用 tokenizer.pad 得到 input_ids + attention_mask
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 把 attention_mask!=1 的位置设为 label_pad_token_id（通常是 -100）
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.label_pad_token_id)

        # 逐样本检查并按样本去除开头 bos（如果确实存在）
        bos_id = self.processor.tokenizer.bos_token_id
        new_labels = []
        for i in range(labels.size(0)):
            l = labels[i]
            # 去除前缀时要先去掉 pad (-100) 再判断
            # 找到第一个非 -100 index
            nonpad_idxs = (l != self.label_pad_token_id).nonzero(as_tuple=False)
            if nonpad_idxs.numel() == 0:
                # 全是 pad
                new_labels.append(torch.tensor([], dtype=torch.long))
                continue
            first_idx = int(nonpad_idxs[0].item())
            # 如果第一个有效 token 是 bos，则去掉它
            if l[first_idx].item() == bos_id:
                # 取从 first_idx+1 开始直到最后的有效 token
                valid = l[first_idx+1:]
                # 过滤掉 trailing pads (-100)
                valid = valid[valid != self.label_pad_token_id]
                new_labels.append(valid.clone().detach())
            else:
                valid = l[first_idx:]
                valid = valid[valid != self.label_pad_token_id]
                new_labels.append(valid.clone().detach())

        # 重新 pad 回 batch，并用 label_pad_token_id 填充
        if len(new_labels) == 0:
            padded = torch.full((0, 0), fill_value=self.label_pad_token_id, dtype=torch.long)
        else:
            max_len = max([int(x.size(0)) for x in new_labels])
            padded = torch.full((len(new_labels), max_len), fill_value=self.label_pad_token_id, dtype=torch.long)
            for i, nl in enumerate(new_labels):
                if nl.numel() > 0:
                    padded[i, : nl.numel()] = nl

        batch["labels"] = padded
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
            # 增加 optimizer 存在性检查
            if hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                self.lrs.append(current_lr)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            self.eval_losses.append((state.global_step, metrics["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        # 绘图逻辑保持不变
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label="Train Loss")
        eval_steps, eval_losses = zip(*self.eval_losses) if self.eval_losses else ([], [])
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="Eval Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss over Steps")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        min_len = min(len(self.steps), len(self.lrs))
        plt.plot(self.steps[:min_len], self.lrs[:min_len], label="Learning Rate")
        plt.xlabel("Global Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Steps")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "lr_curve.png"))
        plt.close()

class AGEMtrainer(Seq2SeqTrainer):
    def __init__(self, replay_buffer, protected_languages, buffer_batch_size_per_task, physical_buffer_batch_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer  # 现在是字典
        self.protected_languages = protected_languages
        self.buffer_batch_size_per_task = buffer_batch_size_per_task
        self.physical_buffer_batch_size = physical_buffer_batch_size
        
        total_samples = len(self.protected_languages) * self.buffer_batch_size_per_task
        print(f"✅ UGP Trainer initialized for fair comparison!")
        print(f"   - Replay data per step: {total_samples} samples ({len(self.protected_languages)} languages x {self.buffer_batch_size_per_task} samples)")

    
    # ### FIX: 重新引入稳健的梯度工具函数 ###
    def _get_grad_vector(self):
        """获取模型可训练参数的梯度并展平为一维向量，处理 None 梯度。"""
        grad_parts = []
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    grad_parts.append(torch.zeros_like(p).view(-1))
                else:
                    grad_parts.append(p.grad.detach().view(-1))
        return torch.cat(grad_parts)

    def _set_grad_vector(self, grad_vector):
        """将一维梯度向量写回模型参数。"""
        pointer = 0
        for p in self.model.parameters():
            if p.requires_grad:
                num_param = p.numel()
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(grad_vector[pointer:pointer + num_param].view_as(p))
                pointer += num_param

    def _project_to_half_space(self, grad, ref_grad):
        """执行梯度投影，包含数值稳定性检查。"""
        # 确保在同一设备上进行计算
        ref_grad_device = ref_grad.to(grad.device)
        
        dot_product = torch.dot(grad, ref_grad_device)
        ref_norm_sq = torch.dot(ref_grad_device, ref_grad_device)
        
        # 增加一个小的 epsilon 来防止除以零
        epsilon = 1e-8
        if dot_product < 0 and ref_norm_sq > epsilon:
            # 投影公式
            grad = grad - (dot_product / (ref_norm_sq + epsilon)) * ref_grad_device
        return grad
    
    # # ### FIX: 高效的参考梯度计算 ###
    # def calculate_reference_gradient(self):
    #     """
    #     从回放缓冲区高效地计算单一的参考梯度。
    #     1. 从缓冲区采样，构建一个混合批次。
    #     2. 对该批次执行一次前向和反向传播。
    #     3. 返回计算得到的梯度向量。
    #     """
    #     if not self.replay_buffer:
    #         return None
        
    #     # 1. 从回放缓冲区采样一个批次
    #     replay_samples = random.sample(self.replay_buffer, min(self.buffer_batch_size, len(self.replay_buffer)))
        
    #     # 2. 对这个批次执行一次前向和反向传播
    #     self.model.zero_grad()
        
    #     # 使用 Trainer 内部的 data_collator
    #     batch = self.data_collator(replay_samples)
    #     # 将数据移动到正确的设备
    #     batch = self._prepare_inputs(batch)

    #     # with torch.no_grad() is wrong, we need grads
    #     outputs = self.model(**batch)
    #     loss = outputs.loss
        
    #     # 使用 Trainer 的方式进行反向传播，以兼容 AMP (fp16) 等
    #     self.accelerator.backward(loss)
        
    #     ref_grad = self._get_grad_vector()
    #     self.model.zero_grad() # 清理梯度，以免影响下一步
        
    #     return ref_grad
    # 在 AGEMtrainer 类中
# 确保删掉旧的 calculate_reference_gradient 函数

    def calculate_reference_gradient(self):
        """
        [Fair Comparison Version]
        为 UGP 计算参考梯度，但使用和 A-GEM 完全相同的平衡采样策略。
        """
        if not self.replay_buffer:
            return None

        # 1. (新逻辑) 从每个受保护的语言中平衡采样
        replay_samples = []
        for lang in self.protected_languages:
            if lang in self.replay_buffer and self.replay_buffer[lang]:
                samples_for_lang = random.sample(
                    self.replay_buffer[lang],
                    min(self.buffer_batch_size_per_task, len(self.replay_buffer[lang]))
                )
                replay_samples.extend(samples_for_lang)
        
        if not replay_samples:
            return None

        # 2. (旧逻辑不变) 使用梯度累积来处理这些采样出的样本
        total_samples_needed = len(replay_samples)
        self.model.zero_grad()
        
        accumulation_steps = (total_samples_needed + self.physical_buffer_batch_size - 1) // self.physical_buffer_batch_size

        for i in range(accumulation_steps):
            start_idx = i * self.physical_buffer_batch_size
            end_idx = start_idx + self.physical_buffer_batch_size
            micro_batch_samples = replay_samples[start_idx:end_idx]

            if not micro_batch_samples:
                continue
                
            batch = self.data_collator(micro_batch_samples)
            batch = self._prepare_inputs(batch)

            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / accumulation_steps
            
            self.accelerator.backward(loss)

        ref_grad = self._get_grad_vector()
        self.model.zero_grad()
        
        return ref_grad

    # ### FIX: 全新、正确的 training_step 逻辑 ###
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # 1. 计算当前任务的梯度 (g_current)
        # 使用 Trainer 的标准 compute_loss
        loss = self.compute_loss(model, inputs)
        
        # 使用 Trainer 的标准方式进行反向传播
        self.accelerator.backward(loss)
        
        # 2. 保存当前梯度
        g_current = self._get_grad_vector()
        
        # 3. 计算参考梯度 (g_ref)
        # 这个函数是独立的，它会自己处理模型的梯度状态
        g_ref = self.calculate_reference_gradient()
        
        # 4. (关键) 恢复当前任务的梯度到模型中
        # 因为 calculate_reference_gradient 修改了模型的 .grad
        self._set_grad_vector(g_current)
        
        # 5. 如果参考梯度存在，则执行投影
        if g_ref is not None:
            g_projected = self._project_to_half_space(g_current, g_ref)
            
            # 6. 将最终的梯度设置回模型
            self._set_grad_vector(g_projected)
            
        # 7. 返回损失，让 Trainer 的主循环处理 optimizer.step() 等
        return loss.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的参数解析保持不变...
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
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=12, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_steps", default=2000, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--output_dir", default="/data/share/guodong/workspace/models/finetune_whisper_trained/whisper-medium")
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/large_cache")
    parser.add_argument("--early_stopping_patience", default=3, type=int, help="Patience for early stopping.")
    args = parser.parse_args()

    # 模型与processor设置
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)

    # 冻结参数, 仅微调decoder (你的逻辑是正确的)
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    # for param in model.model.decoder.embed_tokens.parameters():
    #     param.requires_grad = False
    # for param in model.model.decoder.embed_positions.parameters():
    #     param.requires_grad = False
    
    # 数据集加载
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    er_data_path = os.path.join(args.processed_data_root, "er_data")
    ds = load_from_disk(main_data_path)
    ds_replay = load_from_disk(er_data_path)
    
    # combined_train_data = concatenate_datasets([ds["train"], ds_replay["train"]]).shuffle(seed=42)
    # combined_test_data = concatenate_datasets([ds["test"], ds_replay["test"]]).shuffle(seed=42)
    combined_train_data = ds["train"]
    combined_test_data = ds["test"]
    # # 创建回放缓冲区
    # buffer_size = 2000 
    # replay_subset = ds_replay["train"].shuffle(seed=42).select(range(buffer_size))
    # replay_buffer_list = list(replay_subset)
    # print(f"🚀 创建了一个大小为 {len(replay_buffer_list)} 的回放缓冲区。")
    print("🔧 [Fair Comparison Mode] Building language-classified replay pool for UGP...")
    replay_buffer_dict = defaultdict(list)

    # 确保你的回放数据里有 'language' 字段
    for sample in ds_replay["train"]:
        lang = sample.get("language", "unknown")
        replay_buffer_dict[lang].append(sample)

    protected_languages = list(replay_buffer_dict.keys())
    print(f"   - Detected replay languages: {protected_languages}")
    print(f"✅ Language-classified replay pool created.")


    # 训练参数设置
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        optim="adamw_torch",
        fp16=args.fp16,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=1.0,
        # <--- 修改：添加早停相关的3个核心参数 ---
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience
    )
    trainer = AGEMtrainer(
        replay_buffer=replay_buffer_dict,         # 传入字典
        protected_languages=protected_languages,    # 传入语言列表
        buffer_batch_size_per_task=3,             # !!! 关键：和 agemold.py 的值保持绝对一致
        physical_buffer_batch_size=4,             # 这个值可以根据UGP的需要调整，它不影响数据总量
        model=model,
        args=training_args,
        train_dataset=ds['train'],                 # UGP 和 A-GEM 都只在当前任务数据上训练
        eval_dataset=ds["test"],                   # 保持和agemold一致
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.tokenizer,
        callbacks=[early_stopping_callback],
    )

    # 你可以继续使用自定义的 optimizer 和 scheduler，但要注意
    # Trainer 默认会创建自己的，如果你要覆盖，需要确保兼容性
    # 这里为了简单，我们先注释掉，让 Trainer 使用默认的 AdamW
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # class ReduceLROnPlateauCallback(TrainerCallback):
    #     ...
    # trainer.optimizer = optimizer
    # trainer.add_callback(ReduceLROnPlateauCallback(scheduler))
    
    trainer.add_callback(LoggingCallback(trainer))
    forced = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced

    # ######调试代码
    # # Debug: 检查 datacollator 输出
    # dc = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # from torch.utils.data import DataLoader
    # dl = DataLoader(combined_train_data, batch_size=4, collate_fn=dc, shuffle=False)
    # batch = next(iter(dl))
    # print("INPUTS keys:", list(batch.keys()))
    # print("input_features shape:", batch['input_features'].shape if 'input_features' in batch else 'N/A')
    # print("labels shape:", batch['labels'].shape)
    # labels = batch['labels']
    # print("labels sample (first row):", labels[0][:50])
    # print("非 -100 的比例:", float((labels != -100).sum())/float(labels.numel()))

    
    # 训练模型
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型和处理器
    processor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    print("noer agemnew large训练完成！")