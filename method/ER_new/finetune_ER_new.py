import torch
import argparse
import os

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from load_datasets import load_process_datasets
from transformers import EarlyStoppingCallback, TrainerCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from datasets import concatenate_datasets
from datasets import load_from_disk


# 数据收集器
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

         # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 返回批处理后的数据
        batch["labels"] = labels
        
        # print("features: ", features[0])
        
        if "language" in features[0]:
            batch["language"] = [f["language"] for f in features]
        
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
       #  plt.plot(self.steps, self.lrs, label="Learning Rate")
        plt.xlabel("Global Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Steps")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "lr_curve.png"))
        plt.close()
        
class WeightedLossTrainer(Seq2SeqTrainer):
    def __init__(self, *args, target_langs=None, weight_factor=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_langs = target_langs
        self.weight_factor = weight_factor
    
    # 重写loss计算
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        languages = inputs.get("language")
        
        # 丢掉language标签以符合输入
        inputs.pop('language', None)
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Cross-Entorpy
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        
        loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.size())
        
        # 根据语言分配权重
        weights = torch.ones(labels.size(0), device=labels.device)
        # print(languages)
        for idx, lang in enumerate(languages):
            if lang in self.target_langs:
                weights[idx] = self.weight_factor
                
        loss = (loss.mean(dim=1) * weights).mean()
        
        return (loss, outputs) if return_outputs else loss
        
        
# 加载模型和处理器
def load_model_and_processor(model_name_or_path: str, language: str, task: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    return model, processor

# 冻结编码器参数
def freeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("Encoder parameters are frozen.")

# 设置训练参数
def setup_training_args(args):
    return Seq2SeqTrainingArguments(
        output_dir=f"/data/share/guodong/workspace/models/finetune_whisper_trained/large_er98_v2",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # --- 早停所需参数 ---
        evaluation_strategy="steps",          # 必须按步数评估
        eval_steps=args.eval_steps,           # 保持你原有的评估步数
        # save_steps=args.save_steps,           # 确保与 eval_steps 兼容 (倍数关系)
        load_best_model_at_end=True,          # **必须开启**
        metric_for_best_model="eval_loss",    # 使用验证集损失作为指标
        greater_is_better=False,              # 损失越小越好
        save_total_limit=3,                   # 可选：最多保存几个 checkpoint，节省空间
        # --------------------
        # max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        # evaluation_strategy="steps",
        optim="adamw_torch",
        fp16=args.fp16,
        dataloader_num_workers=16,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        save_steps=args.save_steps,
        # eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
    )

# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Model Fine-tuning")
    parser.add_argument("--model_id", default="small", type=str)
    parser.add_argument("--dataset_root", default="/mnt/lv3/renziang/fleurs2")
    parser.add_argument("--json_output_dir", default="/mnt/lv3/renziang/json_fleurs")
    parser.add_argument("--task", default="transcribe", type=str)
    parser.add_argument("--language", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--max_new_tokens", default=225, type=int)
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=50.0, type=float)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--train_batch_size", default=3, type=int)
    parser.add_argument("--eval_batch_size", default=3, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup_steps", default=2500, type=int)
    # parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--num_train_epochs", default=6, type=int)
    parser.add_argument("--save_steps", default=2000, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--replay_ratio", default=0.2, type=float)
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/large_cache")
    # parser.add_argument("--processed_data_root",  default="/mnt/lv3/renziang/fleurs_cache")


    args = parser.parse_args()

    # 加载模型和处理器
    model_name_or_path = "/data/share/guodong/workspace/models/whisper_hf/whisper-large-v3"
    # model_name_or_path = "/home/chenkaizhe/whisper_finetune-master/examples/asr/logs/whisper-small-freezeencoder-experiment_multilingual_B"
    model, processor = load_model_and_processor(model_name_or_path, args.language, args.task)

    # 冻结编码器参数
    freeze_encoder(model)
    
    # # 冻结Learned Token Embeddings
    # for param in model.model.decoder.embed_tokens.parameters():
    #   param.requires_grad = False

    # # 冻结Learned Positional Encodings
    # for param in model.model.decoder.embed_positions.parameters():
    #     param.requires_grad = False

    # 数据集设置
    datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
    ]
    
    # ER 数据集
    ER_datasets = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]

    # 加载数据集
    print(f"🚀 正在从 {args.processed_data_root} 加载预处理好的数据集...")
    '''ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=args.max_input_length,
        json_output_dir=args.json_output_dir,
        dataset_root=args.dataset_root,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        augment_data = 2,
    )
    
    # print("ds train: ", ds["train"])
    
    # 加载回放数据集
    ds_ER = load_process_datasets(
        ER_datasets,
        processor,
        max_input_length=args.max_input_length,
        dataset_root=args.dataset_root,
        json_output_dir=args.json_output_dir,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        replay_ratio=args.replay_ratio,
        augment_data = 2,
    )

    
    print("训练集数据量：", len(ds["train"]))
    print("测试集数据量：", len(ds["test"]))
    print("ER样本训练集量: ", len(ds_ER["train"]))
    print("ER样本测试集量: ", len(ds_ER["test"]))'''
    

    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    er_data_path = os.path.join(args.processed_data_root, "er_data")
    ds = load_from_disk(main_data_path)
    ds_ER = load_from_disk(er_data_path)
    print("独立数据集加载成功！")
    # 合并数据集
    combined_train_data = concatenate_datasets([ds["train"], ds_ER["train"]]).shuffle(seed=42)
    combined_test_data = concatenate_datasets([ds["test"], ds_ER["test"]]).shuffle(seed=42)


    # 设置训练参数
    training_args = setup_training_args(args)
    
    # 初始化训练器, 目标语种加入weight
    trainer = WeightedLossTrainer(
        args=training_args,
        model=model,
        train_dataset=combined_train_data,
        eval_dataset=combined_test_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        # tokenizer=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        target_langs = ['ms_my', 'fil_ph','id_id','jv_id','mi_nz'],
        weight_factor = 1,##普通er
    )
    # --- 关键修改 2: 向 trainer 添加 EarlyStoppingCallback ---
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3)) # 例如，耐心值设为3

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "encoder" in n],
            "lr": 1e-6,  # 为 encoder 设置的学习率
        },
        {
            "params": [p for n, p in model.named_parameters() if "encoder" not in n],
            "lr": args.learning_rate,  # 为其他层设置的学习率 (1e-5)
        },
    ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    # 定义优化器和学习率调度器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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
    print("ERlarge normal erv2")