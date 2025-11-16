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
from collections import defaultdict

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
    def __init__(self, replay_buffer, protected_languages, buffer_batch_size_per_task=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer # 现在这是一个字典
        self.protected_languages = protected_languages # 需要保护的语言列表
        self.buffer_batch_size_per_task = buffer_batch_size_per_task # 每个旧任务的采样数
        print(f"✅ old A-GEM Trainer 初始化成功！")
        print(f"   - 将保护以下语言: {self.protected_languages}")
        print(f"   - 每个保护语言的批次大小: {self.buffer_batch_size_per_task}")
    
    
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
        if dot_product < 0 and ref_norm_sq > 1e-12: 
            grad = grad - (dot_product / ref_norm_sq) * ref_grad_device
        return grad
        
    
    # ### FIX: 高效的参考梯度计算 ###
    def _calculate_gradient_for_samples(self, samples):
        """为给定的样本列表计算一次梯度向量。"""
        self.model.zero_grad()
        batch = self.data_collator(samples)
        batch = self._prepare_inputs(batch)
        outputs = self.model(**batch)
        loss = outputs.loss
        self.accelerator.backward(loss)
        grad_vector = self._get_grad_vector()
        self.model.zero_grad()
        return grad_vector

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # 1. 计算当前任务的梯度 (g_current)
        self.model.zero_grad() # 确保梯度是干净的
        loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        g_current = self._get_grad_vector()
        self.model.zero_grad() # 清除当前梯度，以便后续计算参考梯度

        # --- 高效的逐语言迭代投影逻辑 ---

        # 2. 遍历每个要保护的旧语言/任务
        for lang in self.protected_languages:
            if lang not in self.replay_buffer or not self.replay_buffer[lang]:
                continue # 如果某个语言没有回放样本，则跳过

            # 2a. 从该语言的回放池中采样
            task_samples = random.sample(
                self.replay_buffer[lang],
                min(self.buffer_batch_size_per_task, len(self.replay_buffer[lang]))
            )

            # 2b. 为这个旧任务的批次计算参考梯度 g_ref_task
            # 注意：这个梯度计算完后只在循环的本次迭代中使用
            g_ref_task = self._calculate_gradient_for_samples(task_samples)

            # 3. (关键) 直接将 g_current 投影到 g_ref_task 定义的半空间
            dot_product = torch.dot(g_current, g_ref_task)
            if dot_product < 0:
                ref_norm_sq = torch.dot(g_ref_task, g_ref_task)
                # 添加一个小的 epsilon 防止除以零
                if ref_norm_sq > 1e-12:
                    projection_scale = dot_product / ref_norm_sq
                    g_current = g_current - projection_scale * g_ref_task
            
            # g_ref_task 在这里作用域结束，其占用的显存会被自动回收

        # 4. 将最终被多次投影后的梯度设置回模型
        self._set_grad_vector(g_current)

        # 5. 返回损失，让 Trainer 的主循环处理 optimizer.step() 等
        return loss.detach()
    # ^^^^^^^^^^^^^^  修改结束 ^^^^^^^^^^^^^^

    def create_optimizer(self):
        """
        重写此方法以创建具有差异化学习率的优化器。
        """
        print("🔧 Creating optimizer with differential learning rates...")
        
        # 从 self.args 中获取全局学习率
        # 注意：这里的 learning_rate 来自 TrainingArguments
        global_lr = self.args.learning_rate 
        encoder_lr = 1e-6 # 你可以硬编码，也可以通过参数传入

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "encoder" in n and p.requires_grad],
                "lr": encoder_lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "encoder" not in n and p.requires_grad],
                "lr": global_lr, 
            },
        ]

        # 创建优化器
        # self.args.optim 来自 TrainingArguments，这里是 "adamw_torch"
        # 其他优化器参数也可以从 self.args 中获取
        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        print(f"✅ Optimizer created. Encoder LR: {encoder_lr}, Decoder LR: {global_lr}")
        return self.optimizer
    # ^^^^^^^^^^^^^^ 新增此方法 ^^^^^^^^^^^^^^^


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的参数解析保持不变...
    parser.add_argument("--model_id", default="/share/guodong/workspace/renziang/whisper_hf/whisper-large-v3")
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
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--train_batch_size", default=6, type=int)
    parser.add_argument("--eval_batch_size", default=6, type=int)
    parser.add_argument("--num_train_epochs", default=4, type=int)
    parser.add_argument("--warmup_steps", default=2500, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--output_dir", default="/data/share/guodong/workspace/models/finetune_whisper_trained/large_agem99_v2")
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/large_cache")
    parser.add_argument("--early_stopping_patience", default=3, type=int, help="Patience for early stopping.")
    args = parser.parse_args()

    # 模型与processor设置
    # model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id, device_map="auto")
    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)
    model.gradient_checkpointing_enable()
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
    
    print("🔧 newnewnew正在为教科书式A-GEM构建按语言分类的回放池...")
    replay_buffer_dict = defaultdict(list)

    # 假设你的 ds_replay 数据集中有一个字段可以标识语言，例如 'language'
    # 遍历回放数据集，按语言进行分组
    for sample in ds_replay["train"]:
        lang = sample.get("language", "unknown") # 请根据你的数据结构修改这里的 'language'
        replay_buffer_dict[lang].append(sample)

    # 确定要保护的语言，这里我们假设保护所有在回放池中找到的语言
    # 你也可以手动指定一个列表，例如 protected_languages = ['en', 'de', 'fr', 'es']
    protected_languages = list(replay_buffer_dict.keys()) 
    print(f"检测到并计划保护的回放语言: {protected_languages}")

    for lang in protected_languages:
        print(f"  - 语言 '{lang}' 的回放样本数: {len(replay_buffer_dict[lang])}")

    print(f"✅ 按语言分类的回放池构建完成，共包含 {len(protected_languages)} 个语言。")


    # 训练参数设置
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        #evaluation_strategy="steps",
        optim="adamw_bnb_8bit", # <--- 修改: 使用 8-bit AdamW 优化器
        fp16=args.fp16,          # <-- 8-bit AdamW 通常与 fp16/bf16 配合使用
        
        gradient_checkpointing=True, # <--- 新增: 告知 Trainer 启用梯度检查点
        # optim="adamw_torch",
        # fp16=args.fp16,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        evaluation_strategy="steps",
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
        replay_buffer=replay_buffer_dict,     # 传入新的字典
        protected_languages=protected_languages, # 传入要保护的语言列表
        buffer_batch_size_per_task=3,        # 每个语言采样4个样本 (可以设为超参数)
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.tokenizer,
        callbacks=[early_stopping_callback],
    )
    # trainer = AGEMtrainer(
    #     replay_buffer=replay_buffer_list,
    #     # 这里 replay_batch_size 应该大一些以获得稳定的梯度
    #     buffer_batch_size=24,
    #     model=model,
    #     args=training_args,
    #     train_dataset=ds['train'],
    #     eval_dataset=ds_replay["test"],
    #     data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    #     # FIX: 传入正确的 tokenizer
    #     tokenizer=processor.tokenizer,
    #     # <--- 修改：将回调函数列表传给 Trainer ---
    #     callbacks=[early_stopping_callback],
    # )

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
    print("large agem old99v2训练完成！")