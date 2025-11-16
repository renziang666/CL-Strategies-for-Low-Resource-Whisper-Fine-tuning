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
class PPAP_EWC:
    def __init__(self, model, ppap_scores_path, device, alpha=0.9):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.model_to_use = model.module if hasattr(model, 'module') else model

        # 1. 加载预先计算好的PPAP分数作为Fisher权重
        print(f"Loading PPAP scores from: {ppap_scores_path}")
        if not os.path.exists(ppap_scores_path):
            raise FileNotFoundError(f"PPAP scores file not found at {ppap_scores_path}")
        
        # 加载分数并立即移动到正确的设备
        self.fisher = {k: v.to(self.device) for k, v in torch.load(ppap_scores_path).items()}
        print("PPAP scores loaded successfully.")

        # 2. 存储当前模型参数作为“旧任务”的参数，用于计算参数变化
        self.params = {}
        for name, param in self.model_to_use.named_parameters():
            if param.requires_grad:
                self.params[name] = param.clone().detach().to(self.device)
        print("Saved current model parameters as old task parameters.")

    # 定义PPAP的loss计算
    def compute_ewc_loss(self, model, lamb=1.0):
        loss = 0.0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # print(name in self.params)
            fisher = self.fisher[name]
            old_param = self.params[name]
            
            if name in self.params:
                if "embed_tokens.weight" in name:
                    diff = param.shape[0] - old_param.shape[0]
                    old_param[name] = torch.nn.functional.pad(old_param[name], [0,0,0,diff])
                    fisher[name] = torch.nn.functional.pad(fisher[name], [0,0,0,diff])
            
                loss += torch.sum(self.fisher[name] * (param - self.params[name]) ** 2) / 2
            
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
    parser.add_argument("--model_id", default="/home/chenkaizhe/whisper_finetune-master/whisper-small", type=str)
    ##待传入
    parser.add_argument("--ppap_scores_path", default="./whisper-small-en-ppap/ppap_scores.pt", type=str, help="Path to the pre-computed PPAP scores file.")
    parser.add_argument("--dataset_root", default="/mnt/lv3/renziang/fleurs2")
    parser.add_argument("--task", default="transcribe", type=str)
    parser.add_argument("--language", default="id", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=50.0, type=float)
    parser.add_argument("--max_new_tokens", default=225, type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=10, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--ewc_lambda", default=5.0, type=float)  # 旧任务loss的权重
    parser.add_argument("--ewc_alpha", default=0.5, type=float)   # fisher矩阵更新率
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=400, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--processed_data_root",  default="/mnt/lv3/renziang/fleurs_cache")

    # 数据集设置
    datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        # ["fleurs", {"language_abbr": "fil_ph"}],
        # ["fleurs", {"language_abbr": "id_id"}],
        # ["common_voice", {"language_abbr": "id_id"}],
    ]
    
    # EWC 数据集 <<< 提示：这个数据集现在只用于“经验回放”，不再用于计算Fisher >>>
    EWC_datasets = [
        # ["fleurs", {"language_abbr": "en_us"}],
        # ["fleurs", {"language_abbr": "zh_cn"}],
        # ["common_voice", {"language_abbr": "zh-CN"}],
        ["fleurs", {"language_abbr": "th_th"}],
        # ["fleurs", {"language_abbr": "vi_vn"}],
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
        
"""    ds = load_process_datasets(
        datasets_settings,
        processor,
        dataset_root=args.dataset_root,
        max_input_length=args.max_input_length,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        augment_data=True,
    )"""
    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    ds = load_from_disk(main_data_path)
    
    # 加载旧任务数据集
"""    ds_EWC = load_process_datasets(
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
        output_dir="/mnt/lv3/renziang/whisper_finetune/EWC",
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
    ewc = PPAP_EWC(
        model=model,
        ppap_scores_path=args.ppap_scores_path, # 传入PPAP分数文件路径
        device=args.device
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
