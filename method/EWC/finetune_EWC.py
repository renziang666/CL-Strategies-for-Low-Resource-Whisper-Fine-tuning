import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0，1，2"

import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from load_datasets import load_process_datasets
from transformers import EarlyStoppingCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from datasets import concatenate_datasets
from datasets import load_from_disk

    
# 数据收集器
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
        # print("labels: ", labels)
        
        return batch

# Fisher 信息存储
class EWC:
    def __init__(self, model, dataset, processor, device, alpha=0.9):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.model_to_use = model.module if hasattr(model, 'module') else model
        
        # 创建batch_size=1的dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # 确保batch_size=1
            shuffle=False,
            collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        )
        
        self.params, self.fisher = self.compute_ewc_params()

    def compute_ewc_params(self):
        self.model.eval()
        self.model.to(self.device)
        
        params = {}
        fisher = {}
        
        # 初始化参数和 Fisher 矩阵（与模型同设备）
        for name, param in self.model_to_use.named_parameters():
            if param.requires_grad:
                params[name] = param.clone().detach().to(self.device)
                fisher[name] = torch.zeros_like(param).to(self.device)
        
        # 计算 Fisher 矩阵
        for i, batch in enumerate(self.dataloader):
            batch = {
                "input_features": batch["input_features"].to(self.device),
                "labels": batch["labels"].to(self.device),
            }
            
            # 确保维度正确
            if batch["input_features"].dim() == 2:
                batch["input_features"] = batch["input_features"].unsqueeze(0)
            if batch["labels"].dim() == 1:
                batch["labels"] = batch["labels"].unsqueeze(0)
            
            # 计算梯度
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # 累加梯度平方（直接在 GPU 上操作，无需 .cpu()）
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += (param.grad ** 2)  # 移除 .cpu()
            
            print(f"Processed {i+1}/{len(self.dataloader)} samples")
        
        # 平均 Fisher 矩阵
        for name in fisher:
            fisher[name] /= len(self.dataloader)
        
        return params, fisher

    # 更新fisher矩阵
    def update_fisher(self, new_fisher):
        for name in self.fisher:
            self.fisher[name] = self.alpha * self.fisher[name] + (1 - self.alpha) * new_fisher[name]
    
    # 定义loss计算
    def compute_ewc_loss(self, model, lamb=1.0):
        loss = 0.0
        for name, param in model.named_parameters():
            # 如果参数当前不可训练，就跳过
            if not param.requires_grad:
                continue
            
            # 关键修复：检查该参数是否在 fisher 字典中
            # 如果它在第一阶段被冻结了，这里就不存在，我们就跳过它
            if name in self.fisher:
                fisher = self.fisher[name]
                old_param = self.params[name]
                loss += torch.sum(fisher * (param - old_param).pow(2)) / 2
        
        return lamb * loss

# 自定义 Trainer 以整合 EWC
class EWCTrainer(Seq2SeqTrainer):
    def __init__(self, ewc: EWC, ewc_lambda=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc
        self.ewc_lambda = ewc_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the original loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Compute the EWC loss
        ewc_loss = self.ewc.compute_ewc_loss(model, self.ewc_lambda)
        
        # Ensure both losses are tensors
        loss_m = loss.mean() if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=loss.device)
        ewc_loss_m = ewc_loss.mean() if isinstance(ewc_loss, torch.Tensor) else torch.tensor(ewc_loss, device=ewc_loss.device)
        
        # print("EWC loss: ", ewc_loss_m)
        
        # Compute the total loss as a tensor
        total_loss = loss_m + ewc_loss_m
        
        # Return the total loss and outputs if requested        
        return (total_loss, loss) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Perform the forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the loss and logits from the outputs
        loss = outputs.loss if hasattr(outputs, "loss") else None
        logits = outputs.logits if hasattr(outputs, "logits") else None

        # Handle the case where logits is None
        if logits is None and isinstance(outputs, tuple):
            logits = outputs[1]  # Assume logits are the second element in the tuple

        # Handle the case where loss is None
        if loss is None and isinstance(outputs, tuple):
            loss = outputs[0]  # Assume loss is the first element in the tuple

        # Prepare the return values
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return (loss, logits, inputs["labels"])

# 冻结编码器参数
def freeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("Encoder parameters are frozen.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Fine-tuning with EWC")
    parser.add_argument("--model_id", default="/mnt/lv2/FLEURS2/EWC_one", type=str)
    parser.add_argument("--dataset_root", default="/mnt/lv3/renziang/fleurs2")
    parser.add_argument("--json_output_dir", default="/mnt/lv3/renziang/json_fleurs")
    parser.add_argument("--task", default="transcribe", type=str)
    parser.add_argument("--language", default="id", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=50.0, type=float)
    parser.add_argument("--max_new_tokens", default=225, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=10, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--ewc_lambda", default=5.0, type=float)  # 旧任务loss的权重
    parser.add_argument("--ewc_alpha", default=0.5, type=float)   # fisher矩阵更新率
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=1200, type=int)
    parser.add_argument("--eval_steps", default=400, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--processed_data_root",  default="/mnt/lv3/renziang/fleurs_cache")

    # 数据集设置
    datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
    ]
    
    # EWC 数据集
    EWC_datasets = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]
    
    # 加载模型和处理器
    torch.backends.cudnn.enabled = False
    args = parser.parse_args()
    
    model, processor = WhisperForConditionalGeneration.from_pretrained(args.model_id), WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)
    
    # 冻结编码器参数
    freeze_encoder(model)
    
    # 冻结Learned Token Embeddings
    for param in model.model.decoder.embed_tokens.parameters():
        param.requires_grad = False

    # 冻结Learned Positional Encodings
    for param in model.model.decoder.embed_positions.parameters():
        param.requires_grad = False

    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    er_data_path = os.path.join(args.processed_data_root, "er_data")
    ds = load_from_disk(main_data_path)
    ds_EWC = load_from_disk(er_data_path)
    print("独立数据集加载成功！")    
    """ds = load_process_datasets(
        datasets_settings,
        processor,
        dataset_root=args.dataset_root,
        json_output_dir=args.json_output_dir,
        max_input_length=args.max_input_length,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        augment_data=True,
    )
    
    # 加载旧任务数据集
    ds_EWC = load_process_datasets(
        EWC_datasets,
        processor,
        max_input_length=args.max_input_length,
        dataset_root=args.dataset_root,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        replay_ratio=0.2,
        augment_data = False,
    )"""
    
    # 合并数据集
    combined_train_data = concatenate_datasets([ds["train"], ds_EWC["train"]])
    combined_test_data = concatenate_datasets([ds["test"], ds_EWC["test"]])

    # 打乱顺序
    combined_train_data.shuffle(seed=42)
    combined_test_data.shuffle(seed=42)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="/mnt/lv2/FLEURS2/EWCtwo",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        optim="adamw_torch",
        fp16=args.fp16,
        dataloader_num_workers=4,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
    )

    # 初始化EWC时需要传入processor
    ewc = EWC(
        model=model,
        dataset=ds_EWC["train"],
        processor=processor,  # 新增processor参数
        device=args.device,
        alpha=args.ewc_alpha
    )
    
    trainer = EWCTrainer(
        ewc=ewc,
        ewc_lambda=args.ewc_lambda,
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=combined_test_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor,
    )
    
   # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 定义学习率调度器
    class ReduceLROnPlateauCallback(TrainerCallback):
        def __init__(self, scheduler):
            self.scheduler = scheduler
            
        def on_evaluate_end(self, args, state, control, **kwargs):
            eval_loss = state.log_history[-1]["eval_loss"]
            self.scheduler.step(eval_loss) 
        
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.add_callback(ReduceLROnPlateauCallback(scheduler))
    
    # 训练模型
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型和处理器
    processor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    print("1003 纯EWC，不冻结")
