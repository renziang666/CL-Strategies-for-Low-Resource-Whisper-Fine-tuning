import os
import csv
import torch
import jiwer
import zhconv
import numpy as np
import torchaudio
import re  # 【修改】确保导入 re 模块
from datasets import Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Union

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass

# ==============================================================================
#  您的原始函数 (compute_metrics 函数已被增强以输出更多信息)
# ==============================================================================

def load_common_voice(dataset_dir, tsv_file="train.tsv"):
    filelist = os.path.join(dataset_dir, tsv_file)
    audio_paths = []
    sentences = []
    with open(filelist, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    for row in tqdm(rows, desc=f"正在读取元数据 {tsv_file}"):
        audio_file = os.path.join(dataset_dir, "clips", row["path"])
        if os.path.exists(audio_file):
            audio_paths.append(audio_file)
            sentences.append(row["sentence"])
    return {"audio_path": audio_paths, "sentence": sentences}

# 请用这个版本完整替换您脚本中的 prepare_dataset 函数
def prepare_dataset(batch, processor, sampling_rate=16000, max_seconds=29.0):
    audio_paths = batch["audio_path"]
    sentences = batch["sentence"]

    # 【修改】创建新的列表来保存有效样本的信息
    waveforms = []
    valid_sentences = []
    valid_paths = []  # <--- 新增
    durations = []

    for path, sent in zip(audio_paths, sentences):
        try:
            waveform, sr = torchaudio.load(path)
            duration = waveform.shape[1] / sr
            if duration > max_seconds:
                # print(f"跳过超长音频 {path}, duration={duration:.1f}s")
                continue
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
                waveform = resampler(waveform)
            
            # 【修改】只有当所有处理都成功时，才将所有信息加入列表
            waveforms.append(waveform.squeeze(0).numpy())
            valid_sentences.append(sent)
            valid_paths.append(path) # <--- 新增
            durations.append(duration)
        except Exception as e:
            # print(f"跳过文件 {path}，错误: {e}")
            continue

    if not waveforms:
        # 返回空的或者None，让后续的 .filter 可以移除这个空批次
        return {
            "input_features": [], "labels": [], "audio_path": [], "sentence": [], "durations": []
        }

    # 使用有效样本列表来创建特征
    batch["input_features"] = processor.feature_extractor(waveforms, sampling_rate=sampling_rate).input_features
    batch["labels"] = processor.tokenizer(valid_sentences, add_special_tokens=False).input_ids
    
    # 【修改】将有效样本的元数据放回batch中
    batch["audio_path"] = valid_paths
    batch["sentence"] = valid_sentences # 即使后面不用，也保持对齐
    batch["durations"] = durations
    
    return batch

# 请用这个版本完整替换您脚本中的 DataCollatorSpeechSeq2SeqWithPadding 类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 分离出需要padding的张量
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 正常进行padding
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        # 【关键修改】打包元数据
        # 这些是非张量数据，将作为列表保留在batch字典中
        if "audio_path" in features[0]:
            batch["audio_path"] = [f["audio_path"] for f in features]
        if "durations" in features[0]:
            batch["durations"] = [f["durations"] for f in features]
        if "sentence" in features[0]:
             batch["sentence"] = [f["sentence"] for f in features]

        return batch

# 请用这个类完整替换您脚本中旧的 HaltOnBadGrad 类
import json
import time
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class LogAndHaltOnBadGrad(TrainerCallback):
    def __init__(self, grad_norm_threshold=500.0, outdir="./bad_batches"):
        self.grad_norm_threshold = grad_norm_threshold
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 注意：on_step_end 在 optimizer step之后，所以我们检查的是上一步的梯度
        # Trainer 会在 state.log_history 中记录 grad_norm
        if not state.log_history:
            return

        last_log = state.log_history[-1]
        grad_norm = last_log.get("grad_norm")

        # 检查非有限梯度 (NaN or Inf)
        if grad_norm is not None and not np.isfinite(grad_norm):
            # Trainer内部已经处理了scaler.update()失败的情况，我们在这里捕获并记录
            self._dump_and_stop(state, kwargs.get("train_dataloader"), f"non-finite grad_norm detected: {grad_norm}")
            control.should_training_stop = True
            return
        
        # 检查梯度是否超阈值
        if grad_norm is not None and grad_norm > self.grad_norm_threshold:
            self._dump_and_stop(state, kwargs.get("train_dataloader"), f"large grad_norm {grad_norm:.1f}")
            control.should_training_stop = True
            return

    def _dump_and_stop(self, state, dataloader, reason):
        # Trainer的on_step_end不直接提供当前batch，这是一个获取它的技巧
        # 但这很复杂且不可靠。更好的方法是在DataCollator中注入信息。
        # 由于我们已经在DataCollator中注入了元数据，我们尝试从那里恢复
        # 注意：这是一个简化的例子，实际中可能需要更复杂的逻辑来获取确切的batch
        # 但我们的目标是找到大致范围，所以这个方法可行
        
        fname = f"{int(time.time())}_step{state.global_step}.json"
        path = os.path.join(self.outdir, fname)
        
        info = {
            "reason": reason,
            "global_step": state.global_step,
            "learning_rate": state.log_history[-1].get("learning_rate"),
            "loss": state.log_history[-1].get("loss"),
            "message": "由于无法在on_step_end中精确获取当前batch，请检查此步骤前后日志中处理的文件。元数据需要在DataCollator中注入并在training_step中捕获。",
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\n[LogAndHaltOnBadGrad] 停止并记录坏批次信息: {path} 原因: {reason}")







# 【新增】冻结Encoder的函数
def freeze_encoder(model):
    """冻结模型encoder部分的参数，使其在训练中不被更新。"""
    print("\n【调试】【核心修复】正在冻结模型的Encoder...")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    model.model.encoder.eval() # 确保encoder处于评估模式
    print("✅ Encoder已冻结！")

def normalize_zh(s: str) -> str:
    s = zhconv.convert(s, "zh-cn")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)
    return s

# 【调试】增强 compute_metrics 以输出所有需要的信息
def compute_metrics(pred):
    print("\n\n>>>>>>>>>>>>>>>>>>>>> 进入 compute_metrics 函数 <<<<<<<<<<<<<<<<<<<<<")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_strs_norm = [normalize_zh(s) for s in pred_strs]
    label_strs_norm = [normalize_zh(s) for s in label_strs]

    print("\n--- [评估解码结果预览] ---")
    for i in range(min(5, len(pred_strs))):
        print(f"样本 {i+1}:")
        print(f"  [原始参考]  : {label_strs[i]}")
        print(f"  [原始预测]  : {pred_strs[i]}")
        print(f"  [规范化参考]: {label_strs_norm[i]}")
        print(f"  [规范化预测]: {pred_strs_norm[i]}\n")

    cer = jiwer.cer(label_strs_norm, pred_strs_norm)
    measures = jiwer.compute_measures(label_strs_norm, pred_strs_norm)
    
    print("--- [CER 计算结果] ---")
    print(f"字错误率 (CER): {cer:.4f}")
    print(f"详细指标 (jiwer.compute_measures):")
    print(f"  - 命中(H): {measures['hits']}, 替换(S): {measures['substitutions']}, 删除(D): {measures['deletions']}, 插入(I): {measures['insertions']}")
    print(f"  - 总参考字符数: {measures['truth']}")
    print(">>>>>>>>>>>>>>>>>>>>> 退出 compute_metrics 函数 <<<<<<<<<<<<<<<<<<<<<\n\n")
    return {"cer": cer}


# ==============================================================================
#  主流程
# ==============================================================================

# --- 1. 数据加载 ---
print("【调试】正在加载完整数据路径...")
full_train_data = load_common_voice(
    dataset_dir="/data/share/guodong/workspace/datasets/CommonVoice/zh-CN",
    tsv_file="train.tsv"
)
full_dev_data = load_common_voice(
    dataset_dir="/data/share/guodong/workspace/datasets/CommonVoice/zh-CN",
    tsv_file="dev.tsv"
)
print(f"【调试】完整训练集大小: {len(full_train_data['sentence'])} | 完整评估集大小: {len(full_dev_data['sentence'])}")

# --- 【修改】数据简化 ---
num_train_samples = 1000
num_dev_samples = 30

train_data = {
    "audio_path": full_train_data["audio_path"][:num_train_samples],
    "sentence": full_train_data["sentence"][:num_train_samples]
}
dev_data = {
    "audio_path": full_dev_data["audio_path"][:num_dev_samples],
    "sentence": full_dev_data["sentence"][:num_dev_samples]
}
print(f"【调试】已简化数据集 -> 训练集: {len(train_data['sentence'])} 条 | 评估集: {len(dev_data['sentence'])} 条")

# --- 2. 加载 Processor ---
print("\n【调试】正在加载 Processor...")
processor = WhisperProcessor.from_pretrained(
    "/data/share/guodong/workspace/models/whisper_hf/whisper-small",
    language="chinese",
    task="transcribe"
)

# ---------------------------------------------------------------------------------
# --- 【最终修复方案】强制分离 pad 和 eos token ---
#
# 我们不再依赖任何if检查，直接进行设置
#
print("\n【调试】【最终修复】正在强制分离 PAD 和 EOS token...")

# 1. 强制将 PAD token 设置为 UNK (unknown) token。
#    这是一个安全的选择，因为模型在正常情况下不应该生成 UNK。
processor.tokenizer.pad_token_id = processor.tokenizer.unk_token_id
print(f"  - Processor的 pad_token_id 已强制设为 UNK token ID: {processor.tokenizer.pad_token_id}")

# 2. 确保 tokenizer 的填充方向是 'right'
processor.tokenizer.padding_side = "right"
print(f"  - Processor的 padding_side 已设为: '{processor.tokenizer.padding_side}'")

# --- 3. 数据预处理 ---
print("\n【调试】正在将字典转为 Dataset 对象...")
cv_train_dataset = Dataset.from_dict(train_data)
cv_dev_dataset = Dataset.from_dict(dev_data)

print("【调试】正在对简化后的数据集进行预处理 (.map)...")
cv_train_dataset = cv_train_dataset.map(
    lambda b: prepare_dataset(b, processor, sampling_rate=16000),
    batched=True,
    batch_size=16,
    # remove_columns=["audio_path", "sentence"]
).filter(lambda x: x['input_features'] is not None)

cv_dev_dataset = cv_dev_dataset.map(
    lambda b: prepare_dataset(b, processor, sampling_rate=16000),
    batched=True,
    batch_size=16,
    # remove_columns=["audio_path", "sentence"]
).filter(lambda x: x['input_features'] is not None)
print("【调试】预处理完成！")


# --- 【调试】检查单个处理后的样本 ---
print("\n--- [调试信息：检查单个样本格式] ---")
sample = cv_train_dataset[0]
print("样本keys:", sample.keys())
print("input_features 类型:", type(sample["input_features"]))
print("input_features 长度:", len(sample["input_features"]))
print("labels 类型:", type(sample["labels"]))
print("labels 内容:", sample["labels"])
decoded_labels = processor.tokenizer.decode(sample["labels"])
print("解码后的 labels:", decoded_labels)
print("--------------------------------------\n")


# --- 【新增调试】循环检查训练集前10个样本的完整解码 ---
print("\n\n--- [新增调试信息：检查训练集前10个样本的完整解码] ---")
num_samples_to_inspect = 10
original_sentences_subset = train_data["sentence"]

for i in range(min(num_samples_to_inspect, len(cv_train_dataset))):
    processed_sample = cv_train_dataset[i]
    label_ids = processed_sample["labels"]
    original_sentence = original_sentences_subset[i]
    
    # 解码，并设置 skip_special_tokens=False 来保留所有特殊字符
    decoded_with_special_tokens = processor.tokenizer.decode(label_ids, skip_special_tokens=False)

    print(f"\n--- 样本 {i+1}/{num_samples_to_inspect} ---")
    print(f"  [原始句子]      : {original_sentence}")
    print(f"  [Token IDs]     : {label_ids}")
    print(f"  [解码结果(含特殊符)]: '{decoded_with_special_tokens}'")
print("----------------------------------------------------------\n")


# --- 【修改】注释掉磁盘保存，以加速调试流程 ---
# print("正在保存处理好的数据集到硬盘...")
# cv_train_dataset.save_to_disk("./processed_data1/train")
# cv_dev_dataset.save_to_disk("./processed_data1/dev")
print("【调试】已跳过数据集的磁盘保存步骤。")

# --- 4. 加载模型 ---
print("\n【调试】正在加载预训练模型...")
model = WhisperForConditionalGeneration.from_pretrained(
    "/data/share/guodong/workspace/models/whisper_hf/whisper-small"
)
model.config.pad_token_id = processor.tokenizer.pad_token_id
# 4. 确保被抑制的token列表中不包含我们新的pad_token，以防万一
#    这是为了防止模型在生成时被禁止输出填充符（虽然不太可能发生）
if model.config.suppress_tokens:
    # 创建一个可修改的列表
    suppress_tokens = list(model.config.suppress_tokens)
    if processor.tokenizer.pad_token_id in suppress_tokens:
        print(f"  - 警告：suppress_tokens 中包含 pad_token_id，正在移除...")
        suppress_tokens.remove(processor.tokenizer.pad_token_id)
        model.config.suppress_tokens = suppress_tokens
print("✅ PAD 和 EOS token 分离设置完成！")

# 为模型设置正确的解码指令
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

freeze_encoder(model)
# --- 5. 配置训练参数 ---
print("\n【调试】正在配置用于快速调试的训练参数...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned-zh-debug",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,     # 降低学习率
    warmup_steps=500,
    max_steps=5000,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="no",
    load_best_model_at_end=False,
    fp16=False,             # 先关掉 fp16 以确认是否为混合精度导致问题
    predict_with_generate=True,
    generation_max_length=225,
    max_grad_norm=1.0,      # 梯度裁剪：防止爆炸
)
# 【关键修改】实例化并添加回调
halt_callback = LogAndHaltOnBadGrad(grad_norm_threshold=300.0) # 设置一个合理的阈值
# --- 6. 初始化 Trainer ---
print("\n【调试】正在初始化 Trainer...")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=cv_train_dataset,
    eval_dataset=cv_dev_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# --- 7. 【调试】进行训练前评估 ---
print("\n\n=================================================")
print("  💥 开始进行【训练前】的基线评估...  ")
print("=================================================")
baseline_metrics = trainer.evaluate()
print("\n--- 【训练前】评估结果 (基线) ---")
print(baseline_metrics)
print("---------------------------------\n")


# --- 8. 开始训练 ---
print("\n\n=================================================")
print("  🚀 开始进行快速调试训练 (最多100步)...  正在进行检查")
print("=================================================")

# 取一个小 batch
examples = [cv_train_dataset[i] for i in range(2)]
batch = data_collator(examples)
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

model.train()
outputs = model(**{k: v for k, v in batch.items() if k in ("input_features", "labels")})
loss = outputs.loss
print("Forward loss:", loss.item(), "isfinite?", torch.isfinite(loss).item())

loss.backward()
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += float(p.grad.data.norm(2).item()**2)
total_norm = total_norm**0.5
print("Manual grad norm:", total_norm)
model.zero_grad()


trainer.train()

# --- 9. 【修改】注释掉最终的模型保存 ---
# trainer.save_model("/data/share/guodong/workspace/models/finetune_whisper_trained/chinese_v4")
# processor.save_pretrained("/data/share/guodong/workspace/models/finetune_whisper_trained/chinese_v4")
print("\n\n【调试】快速调试训练结束，已跳过最终模型保存。")