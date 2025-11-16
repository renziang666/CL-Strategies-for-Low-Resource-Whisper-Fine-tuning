import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import os

from tqdm.auto import tqdm
import collections
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from load_datasets import load_process_datasets
from transformers import EarlyStoppingCallback, TrainerCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from datasets import concatenate_datasets,load_from_disk,Dataset

from collections import Counter



ENGLISH_LIKE_LANGS = {"en_us", "fr_fr", "id_id", "ms_my", "fil_ph", "jv_id", "mi_nz"}
def _maybe_normalize(text: str, lang_code: str):
    if lang_code in ENGLISH_LIKE_LANGS:
        return normalizer(text)
    return text
# 数据收集器
# ✅ 请用这个最终版本完全替换您脚本中现有的 DataCollatorSpeechSeq2SeqWithPadding 类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从 features 中提取 input_features 和 labels
        # input_features 是一个列表，其中每个元素都是一个音频特征数组
        input_features = [feature["input_features"] for feature in features]
        # labels 是一个列表，其中每个元素都是一个 token id 列表
        label_features = [feature["labels"] for feature in features]

        # 对音频特征进行填充(padding)。
        # 这个方法会返回一个包含 'input_features' 和 'attention_mask' 的字典。
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        
        # --- 新增的调试代码 ---
        # 让我们确认一下 `pad` 函数是否真的返回了 attention_mask
        # print(f"DEBUG Collator: batch keys from pad(): {list(batch.keys())}")
        # --------------------

        # 对标签进行填充
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, return_tensors="pt"
        )

        # 将标签中被填充的部分（padding）替换为-100，这样在计算损失时它们会被忽略
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果解码器的起始token（bos_token）在之前的步骤中被添加了，我们在这里移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理好的标签添加到最终的批次中
        batch["labels"] = labels
        
        # 将语言信息也添加到批次中
        if "language" in features[0]:
            batch["language"] = [f["language"] for f in features]

        return batch

import jiwer
import math
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_utils import EvalPrediction

# 创建一个全局的 normalizer，用于清理文本
normalizer = BasicTextNormalizer()

def compute_metrics(pred: EvalPrediction):
    # pred 对象包含两个主要部分: predictions (模型输出的 logits) 和 label_ids (真实标签)
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 将标签中的 -100 (被忽略的token) 替换为 tokenizer 的 pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 将 token ID 解码回文本字符串
    # skip_special_tokens=True 会移除像 <|startoftranscript|> 这样的特殊标记
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # 规范化文本，去除大小写、标点等，使WER计算更公平
    pred_str = [normalizer(text) for text in pred_str]
    label_str = [normalizer(text) for text in label_str]

    # 使用 jiwer 计算WER，乘以100得到百分比形式
    wer = 100 * jiwer.wer(label_str, pred_str)

    return {"wer": wer}

def balance_dataset_by_language(dataset: Dataset) -> Dataset:
    """
    对输入的数据集按 'language' 字段进行下采样，使每个语种的样本数相同。

    Args:
        dataset: 包含 'language' 列的 Hugging Face 数据集。

    Returns:
        一个各语种样本数均衡的新数据集。
    """
    # 1. 统计每个语言的样本数
    lang_counts = Counter(dataset["language"])
    if not lang_counts:
        print("⚠️ 数据集中没有 'language' 信息，无法进行均衡化，将返回原数据集。")
        return dataset
        
    print(f"原始语言分布: {lang_counts}")

    # 2. 找到样本数最少的语言作为目标数量
    min_samples = min(lang_counts.values())
    print(f"确定均衡化目标样本数 (按最少语种): {min_samples}")

    balanced_subsets = []
    # 3. 对每个语种进行下采样
    for lang, count in lang_counts.items():
        # 筛选出当前语言的子集
        lang_subset = dataset.filter(lambda example: example['language'] == lang)
        
        # 打乱后，选取与最少样本数相同的数量
        balanced_subset = lang_subset.shuffle(seed=42).select(range(min_samples))
        balanced_subsets.append(balanced_subset)
        print(f"  - 语种 '{lang}' 已从 {count} 下采样至 {min_samples}")

    # 4. 将所有均衡后的子集重新合并
    final_balanced_dataset = concatenate_datasets(balanced_subsets)
    
    # 最后再对整体进行一次打乱
    return final_balanced_dataset.shuffle(seed=42)



# 请用这个版本完全替换您脚本中现有的 DynamicReplayTrainer 类
# ✅✅✅ 这是最终的解决方案。请用这个类完全替换您代码中的旧版本。
class DynamicReplayTrainer(Seq2SeqTrainer):
    # __init__, rebuild_train_dataset, compute_loss 方法都正确，保持不变
    def __init__(self, *args, processor=None, main_train_dataset=None, er_dataset_pool=None, total_replay_ratio=0.2,main_lang_codes=None, main_data_weight_factor=2,  **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor 
        self.main_train_dataset = main_train_dataset
        self.er_dataset_pool = er_dataset_pool
        self.total_replay_ratio = total_replay_ratio
        self.main_lang_codes = main_lang_codes if main_lang_codes is not None else []
        self.main_data_weight_factor = main_data_weight_factor
        self.er_lang_weights = {
            'en_us': 0.0725,
            'fr_fr': 0.1092,
            'th_th': 0.6506,
            'vi_vn': 0.1677
        }
        print(f"✅ 初始化ER语言权重: {self.er_lang_weights}")
        print(f"✅ 主任务语言将使用 {self.main_data_weight_factor} 的损失权重。")
        self.rebuild_train_dataset()

    def rebuild_train_dataset(self):
        print("\n rebuilding training dataset")
        total_main_samples = len(self.main_train_dataset)
        total_er_samples_to_sample = int(total_main_samples / (1 - self.total_replay_ratio) * self.total_replay_ratio)
        print(f"主任务样本数: {total_main_samples}, 总回放样本数: {total_er_samples_to_sample}")
        er_subsets_to_combine = []
        for lang, weight in self.er_lang_weights.items():
            num_samples_for_lang_calculated = int(total_er_samples_to_sample * weight)
            er_dataset_for_lang = self.er_dataset_pool[lang]
            num_available_samples = len(er_dataset_for_lang)
            num_samples_to_draw = min(num_samples_for_lang_calculated, num_available_samples)
            if num_samples_to_draw == 0:
                print(f"  - 语言'{lang}' (权重 {weight:.2f}): 计算出需采样0个样本，已跳过。")
                continue
            sampled_subset = er_dataset_for_lang.shuffle(seed=random.randint(0, 10000)).select(range(num_samples_to_draw))
            er_subsets_to_combine.append(sampled_subset)
            print(f"  - 语言'{lang}' (权重 {weight:.2f}): 计划采样 {num_samples_for_lang_calculated}，实际可用 {num_available_samples}，最终采样 {num_samples_to_draw} 个样本")
        new_train_dataset = concatenate_datasets([self.main_train_dataset] + er_subsets_to_combine)
        self.train_dataset = new_train_dataset.shuffle(seed=42)
        print(f"✅ 新的训练集构建完成，总样本数: {len(self.train_dataset)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        languages = inputs.pop("language", None)
        # 如果attention_mask不存在，模型也能处理，只是效率较低
        loss_inputs = {"input_features": inputs["input_features"], "labels": inputs["labels"]}
        if "attention_mask" in inputs:
            loss_inputs["attention_mask"] = inputs["attention_mask"]
            
        outputs = model(**loss_inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)
        weights = torch.ones(labels.size(0), device=labels.device)
        if languages and self.main_lang_codes:
            for i, lang in enumerate(languages):
                if lang in self.main_lang_codes:
                    weights[i] = self.main_data_weight_factor
        weighted_loss = (loss.mean(dim=1) * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    # ✅ 增加了tqdm进度条的最终版 evaluation_loop
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: List[str] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        results_by_lang = collections.defaultdict(lambda: {"refs": [], "preds": []})
        all_refs = []
        all_preds = []
        
        print(f"\n🔬 开始内存优化的评估循环 ({description})...")
        # ✅ 使用 tqdm 包装 dataloader 以显示进度条
        for step, inputs in enumerate(tqdm(dataloader, desc="Evaluating")):
            languages = inputs.pop("language", None)
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_features=inputs["input_features"], 
                    attention_mask=inputs.get("attention_mask"),
                    max_length=self.args.generation_max_length,
                    num_beams=self.args.generation_num_beams,
                ).cpu().numpy()

                labels = inputs["labels"].cpu().numpy()

            labels[labels == -100] = self.processor.tokenizer.pad_token_id
            decoded_preds = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
            if languages:
                decoded_preds = [_maybe_normalize(p, lang) for p, lang in zip(decoded_preds, languages)]
                decoded_labels = [_maybe_normalize(l, lang) for l, lang in zip(decoded_labels, languages)]
            else:
                decoded_preds = decoded_preds
                decoded_labels = decoded_labels

            if languages:
                for lang, pred, label in zip(languages, decoded_preds, decoded_labels):
                    results_by_lang[lang]["preds"].append(pred)
                    results_by_lang[lang]["refs"].append(label)
            
            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_labels)

        metrics = {}
        total_wer = 100 * jiwer.wer(all_refs, all_preds)
        metrics[f"{metric_key_prefix}_wer"] = total_wer
        print(f"   - ✅ 总体评估WER: {total_wer:.2f}%")


        print(f"   - 正在为ER语言单独计算评估WER和（对 th_th）CER...")
        for lang_code, lang_results in results_by_lang.items():
            if lang_code in self.er_dataset_pool:
                # 避免空列表
                refs = lang_results["refs"]
                preds = lang_results["preds"]
                if len(refs) == 0:
                    print(f"     - 语言'{lang_code}'没有样本，跳过。")
                    continue

                

                # 如果是泰语，额外计算字符级 CER（百分比）
                if lang_code == "th_th":
                    try:
                        lang_cer = 100 * jiwer.cer(refs, preds)
                        metrics[f"{metric_key_prefix}_{lang_code}_wer"] = lang_cer
                        print(f"     - 语言'{lang_code}'的评估CER: {lang_cer:.2f}%")
                    except Exception as e:
                        print(f"     - 计算 {lang_code} CER 时出错: {e}")
                else:
                    # 默认计算 WER
                    lang_wer = 100 * jiwer.wer(refs, preds)
                    metrics[f"{metric_key_prefix}_{lang_code}_wer"] = lang_wer
                    print(f"     - 语言'{lang_code}'的评估WER: {lang_wer:.2f}%")


        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(all_refs))

class DynamicReplayCallback(TrainerCallback):
    def __init__(self, trainer, temperature=1.0):
        self.trainer = trainer
        self.temperature = temperature

    # ✅ 核心升级：on_evaluate 现在读取 WER 而不是 loss
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print("\n🔄 动态回放回调: 评估结束，准备根据WER更新采样权重...")
        if metrics is None:
            return

        er_wers = {}
        # 从 metrics 中提取每个ER语言的WER
        for lang in self.trainer.er_dataset_pool.keys():
            wer_key = f"eval_{lang}_wer"  # <-- 读取WER
            if wer_key in metrics:
                er_wers[lang] = metrics[wer_key]
        
        if not er_wers:
            print("  - 未找到ER语言的单独WER，跳过权重更新。")
            return
            
        print(f"  - 获取到ER语言WER: {er_wers}")

        # # --- 核心数学方法：使用带温度的 Softmax 将WER(越高越差)转换为权重(越高越好) ---
        # langs = list(er_wers.keys())
        # wers = np.array([er_wers[lang] for lang in langs])
        
        # wers += 1e-6 # 防止为0
        
        # scaled_wers = wers / self.temperature
        
        # exp_wers = np.exp(scaled_wers)
        # probabilities = exp_wers / np.sum(exp_wers)
        
        # # 更新 trainer 中的权重
        # new_weights = {lang: prob for lang, prob in zip(langs, probabilities)}

        # ✅✅✅ 方案一：线性比例加权 ✅✅✅
        langs = list(er_wers.keys())
        wers = np.array([er_wers[lang] for lang in langs])
        
        # 防止所有WER都为0的情况
        if np.sum(wers) == 0:
            # 如果所有WER都是0，则平均分配权重
            num_langs = len(langs)
            probabilities = np.full(num_langs, 1.0 / num_langs)
        else:
            # 权重与WER成正比
            probabilities = wers / np.sum(wers)
        
        probabilities[probabilities < 0.1] = 0.1
        probabilities = probabilities / np.sum(probabilities)

        # 更新 trainer 中的权重
        new_weights = {lang: prob for lang, prob in zip(langs, probabilities)}
        self.trainer.er_lang_weights = new_weights
        
        print(f"  - 计算出新的采样权重 (线性比例): {self.trainer.er_lang_weights}")

        # 调用方法，为下一轮训练重建数据集
        self.trainer.rebuild_train_dataset()
        
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
        output_dir=f"/data/share/guodong/workspace/models/finetune_whisper_trained/small_dynamic_er_full_v2",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # --- 早停所需参数 ---
        evaluation_strategy="steps",          # 必须按步数评估
        eval_steps=args.eval_steps,           # 保持你原有的评估步数
        # save_steps=args.save_steps,           # 确保与 eval_steps 兼容 (倍数关系)
        load_best_model_at_end=True,          # **必须开启**
        metric_for_best_model="wer",       # <-- ✅ 正确的位置：监控wer
        greater_is_better=False,         # <-- ✅ 正确的位置：wer越小越好
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
        generation_num_beams=1, # ‼️ 关键：将集束搜索大小设为1，即贪心搜索
    )


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
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=3, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup_steps", default=1500, type=int)
    # parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--replay_ratio", default=0.2, type=float)
    parser.add_argument("--processed_data_root",  default="/data/share/guodong/workspace/datasets/FLEURS/cache")
    parser.add_argument("--main_data_weight", default=2, type=float)
    # parser.add_argument("--processed_data_root",  default="/mnt/lv3/renziang/fleurs_cache")


    args = parser.parse_args()

    # 加载模型和处理器
    model_name_or_path = "/data/share/guodong/workspace/models/whisper_hf/whisper-small"
    # model_name_or_path = "/home/chenkaizhe/whisper_finetune-master/examples/asr/logs/whisper-small-freezeencoder-experiment_multilingual_B"
    model, processor = load_model_and_processor(model_name_or_path, args.language, args.task)

    # # 冻结编码器参数
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
    er_lang_codes = [setting[1]['language_abbr'] for setting in ER_datasets]
    main_lang_codes = [setting[1]['language_abbr'] for setting in datasets_settings]

    # 加载数据集
    print("🚀 正在加载预处理好的独立数据集...")
    main_data_path = os.path.join(args.processed_data_root, "main_data")
    er_data_path = os.path.join(args.processed_data_root, "er_data_dy")
    ds = load_from_disk(main_data_path)
    ds_ER = load_from_disk(er_data_path)
    print("独立数据集加载成功！")
    # ###debug # —— 调试用：仅取前 100 条样本快速跑通流程（测试用，跑通后请删除或注释） —— 
    # debug_n = 100 
    # ds["train"] = ds["train"].select(range(min(debug_n, len(ds["train"])))) 
    # ds["test"] = ds["test"].select(range(min(10, len(ds["test"])))) 
    # ds_ER["train"] = ds_ER["train"].select(range(min(debug_n, len(ds_ER["train"])))) 
    # ds_ER["test"] = ds_ER["test"].select(range(min(10, len(ds_ER["test"])))) 
    # # -----------------------------------------------------------------------------------
    balanced_ds_ER_train = balance_dataset_by_language(ds_ER["train"])
    balanced_ds_ER_test = balance_dataset_by_language(ds_ER["test"])

    # 创建一个ER语言池的字典，方便后续按语言采样
    er_dataset_pool = {
        lang: balanced_ds_ER_train.filter(lambda ex: ex['language'] == lang)
        for lang in er_lang_codes
    }

    combined_eval_dataset = concatenate_datasets([ds["test"], balanced_ds_ER_test])


    # 设置训练参数
    training_args = setup_training_args(args)
    
    # 初始化训练器, 目标语种加入weight
    trainer = DynamicReplayTrainer(
        model=model,
        args=training_args,
        main_train_dataset=ds["train"],   # 传入主训练集
        processor=processor,
        er_dataset_pool=er_dataset_pool,  # 传入ER语言池
        total_replay_ratio=args.replay_ratio,           # 传入总回放率
        eval_dataset=combined_eval_dataset, # 使用合并后的评估集
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        main_lang_codes=main_lang_codes,              # <-- [新增] 传入主任务语言列表
        main_data_weight_factor=args.main_data_weight, # <-- [新增] 传入权重因子
        compute_metrics=compute_metrics, # <-- 新增：将函数传递给Trainer
    )
    # 添加动态回放回调，它会在每次评估后更新采样权重
    trainer.add_callback(DynamicReplayCallback(trainer, temperature=20)) # T值可以调整，越大权重越平均
    
    # 添加您已有的其他回调
    # --- 关键修改 2: 向 trainer 添加 EarlyStoppingCallback ---
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3)) # 例如，耐心值设为3

    
    # 定义优化器和学习率调度器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # class ReduceLROnPlateauCallback(TrainerCallback):
    #    def __init__(self, scheduler):
    #        self.scheduler = scheduler
            
    #    def on_evaluate_end(self, args, state, control, **kwargs):
    #        eval_loss = state.log_history[-1]["eval_loss"]
    #        self.scheduler.step(eval_loss)
    
    # trainer.optimizer = optimizer
    # trainer.add_callback(ReduceLROnPlateauCallback(scheduler))
    # trainer.add_callback(LoggingCallback(trainer))
    
    # 训练模型
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型和处理器
    processor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    print("动态ER medium 1")