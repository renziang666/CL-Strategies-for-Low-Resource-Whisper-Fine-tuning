import os
import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from load_datasets import load_process_datasets # 假设你这个函数可用
from datasets import load_from_disk

# DataCollator 和 LoggingCallback 可以从你的第一阶段脚本中直接复制过来
# (这里为了简洁省略，假设它们已经定义好了)
from finetune_PPAP_stepone import DataCollatorSpeechSeq2SeqWithPadding, LoggingCallback

class ContinualLearningTrainer(Seq2SeqTrainer):
    def __init__(self, ppap_scores, original_params, ewc_lambda=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.device="cuda:0"
        # 💡 关键修改：将加载的 CPU 张量移动到与模型相同的设备上
        # self.device 来自于 Trainer 的 self.args.device，可以确保设备一致
        self.ppap_scores = {name: p.to(self.device) for name, p in ppap_scores.items()}
        self.original_params = {name: p.to(self.device) for name, p in original_params.items()}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 计算当前新任务的标准损失
        loss_new_task, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # 2. 计算 EWC 惩罚项
        loss_ewc = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad: # 只对可训练的参数计算惩罚
                # 获取对应的分数和原始参数
                importance = self.ppap_scores[name]
                original_param = self.original_params[name]
                
                # 计算参数的变化量并施加惩罚
                # EWC Loss = Σ (importance * (current_param - original_param)^2)
                loss_ewc += torch.sum(importance * (param - original_param).pow(2))
        
        # 3. 计算总损失
        total_loss = loss_new_task + self.ewc_lambda * loss_ewc
        
        # 打印loss，方便调试
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"\nStep: {self.state.global_step}, New Task Loss: {loss_new_task.item():.4f}, EWC Loss: {loss_ewc.item():.4f}, Total Loss: {total_loss.item():.4f}")

        return (total_loss, outputs) if return_outputs else total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- 关键路径参数 ---
    parser.add_argument("--base_model_path", default='/mnt/g2/chenkaizhe/whisper_finetune-master/whisper-small', type=str, help="第一阶段训练时使用的、文件完整的基础模型路径")
    parser.add_argument("--stage1_model_path", default='/mnt/lv2/FLEURS2/PPAP1', type=str, help="第一阶段训练好的模型 checkpoint 路径")
    parser.add_argument("--ppap_scores_path", default='/mnt/lv2/FLEURS2/PPAP1/ppap_scores.pt', type=str, help="第一阶段生成的 ppap_scores.pt 文件路径")
    parser.add_argument("--output_dir", default='/home/renziang/output_model/PAPP2', type=str, help="第二阶段模型的输出路径")

    # --- 新任务数据集参数 ---
    parser.add_argument("--new_task_data_path", default="/mnt/lv3/renziang/fleurs_cache/main_data", type=str, help="新任务预处理好的数据缓存路径")
    
    # --- EWC 超参数 ---
    parser.add_argument("--ewc_lambda", default=5000.0, type=float, help="EWC 惩罚项的权重")
    
    # --- 其他训练参数 (可以和第一阶段保持一致或微调) ---
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="ms")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    # ... 其他你需要的训练参数 ...
    
    args = parser.parse_args()

    # --- 修改加载逻辑 ---
    print("--- Loading Stage 1 artifacts ---")
    
    # 2. ✅ 正确的加载方式
    # 从文件完整的基础模型路径加载处理器
    print(f"Loading processor from base model path: {args.base_model_path}")
    processor = WhisperProcessor.from_pretrained(args.base_model_path, language=args.language, task=args.task)
    
    # 从第一阶段的 checkpoint 加载微调好的模型
    print(f"Loading fine-tuned model from Stage 1 checkpoint: {args.stage1_model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(args.stage1_model_path)
    
    
    ppap_scores = torch.load(args.ppap_scores_path, map_location=model.device)
    
    # 存储第一阶段的原始参数，用于计算惩罚
    original_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    
    # 2. 准备新任务的数据集
    print(f"--- Loading new task dataset from {args.new_task_data_path} ---")
    ds_new_task = load_from_disk(args.new_task_data_path)
    
    # (可选) 冻结与第一阶段相同的层
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # 3. 设置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        # ... 其他你需要的 Seq2SeqTrainingArguments 参数 ...
    )

    # 4. 初始化我们自定义的 Trainer
    trainer = ContinualLearningTrainer(
        ppap_scores=ppap_scores,
        original_params=original_params,
        ewc_lambda=args.ewc_lambda,
        args=training_args,
        model=model,
        train_dataset=ds_new_task["train"],
        eval_dataset=ds_new_task["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
    )
    
    # 5. 开始第二阶段的训练
    print("\n--- Starting Stage 2 Continual Learning Training ---")
    trainer.train()

    # 6. 保存最终模型
    print("--- Stage 2 Training Finished. Saving final model. ---")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)