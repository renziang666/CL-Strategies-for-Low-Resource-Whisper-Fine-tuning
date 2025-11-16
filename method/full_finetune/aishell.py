import os
import torch
import gc
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments

# --- 1. 设置 Hugging Face 国内镜像源 ---
# 这是一个关键步骤，确保在国内服务器上可以快速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 设置训练参数和路径 ---
# 指定你希望数据集下载和缓存的位置
# 请确保你的磁盘空间足够，AISHELL-2 数据集很大（几十GB）
DATA_CACHE_DIR = "/data/share/guodong/workspace/datasets/FLEURS/aishell"

# 模型保存路径
OUTPUT_DIR = "/data/share/guodong/workspace/models/finetune_whisper_trained/aishell"

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,  # 根据你的显存大小调整
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    evaluation_strategy="epoch",  # 每跑完一轮 epoch 进行评估
    fp16=True,  # 启用混合精度训练，可以加速并节省显存
    push_to_hub=False,  # 不上传到 Hugging Face Hub
)

# --- 3. 加载数据集 ---
print("正在加载和下载 AISHELL-1 数据集...")
# load_dataset 会将数据集下载到指定的缓存目录
dataset = load_dataset(
    "speechcolab/aishell",
    data_dir="/data/share/guodong/workspace/datasets/FLEURS/aishell/AISHELL-1",
    cache_dir=DATA_CACHE_DIR
)
# 将音频列转换成 16kHz 的单声道，这是 Whisper 模型的标准输入
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# --- 4. 加载模型和处理器 ---
print("正在加载 Whisper 模型和处理器...")
model_name = "/data/share/guodong/workspace/models/whisper_hf/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# 确保 tokenizer 的 pad_token 和 eos_token 设置正确
processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.forced_decoder_ids = None

# --- 5. 数据预处理函数 ---
def prepare_dataset(batch):
    # 加载音频文件并将其重采样到 16kHz
    audio = batch["audio"]

    # 提取音频特征（mel spectrogram）
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # 编码文本转录
    batch["labels"] = processor.tokenizer(
        batch["text"],
        padding=True,
        max_length=processor.tokenizer.model_max_length,
        truncation=True
    ).input_ids

    return batch

# 对数据集应用预处理
print("正在对数据集进行预处理...")
train_dataset = dataset["train"].map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    num_proc=4 # 可以根据你的 CPU 核心数调整
)

eval_dataset = dataset["validation"].map(
    prepare_dataset,
    remove_columns=dataset["validation"].column_names,
    num_proc=4 # 可以根据你的 CPU 核心数调整
)

# 释放内存，因为原始数据集的音频数据比较占内存
del dataset
gc.collect()

# --- 6. 定义 Data Collator ---
# 自动批处理数据
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从特征中提取 input_features 和 labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt").input_ids

        # 将 -100 标记用于不需要计算损失的 token
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 7. 创建 Trainer 并开始训练 ---
print("正在创建 Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

print("开始训练...")
trainer.train()

print("训练完成！")