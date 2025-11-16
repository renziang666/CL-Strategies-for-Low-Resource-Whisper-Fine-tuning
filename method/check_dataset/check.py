# debug_cache.py
import os
from datasets import load_from_disk
from transformers import WhisperProcessor

# --- 请根据你的实际情况修改下面两个路径 ---

# 1. 你处理好的、用于训练的数据缓存文件夹路径
# 从你的训练脚本看，应该是这个路径
PROCESSED_DATA_PATH = "/data/share/guodong/workspace/datasets/FLEURS/chinese_cache_commonvoice_v3/main_data"

# 2. 你训练时使用的 Whisper Processor 的基础模型路径
# 从你的训练脚本看，应该是这个路径
PROCESSOR_BASE_PATH = "/data/share/guodong/workspace/models/whisper_hf/whisper-small"

# --- 下面的代码无需修改 ---

def inspect_data(processed_path, processor_path):
    """
    加载并检查处理好的数据集缓存。
    """
    if not os.path.exists(processed_path):
        print(f"❌ 错误：找不到缓存路径 '{processed_path}'")
        print("请确认路径是否正确，或者你的数据是否已成功生成。")
        return

    print(f"🔍 正在加载 Processor 从: {processor_path}")
    try:
        processor = WhisperProcessor.from_pretrained(processor_path)
    except Exception as e:
        print(f"❌ 错误：加载 Processor 失败: {e}")
        return

    print(f"💿 正在加载已处理的数据集从: {processed_path}")
    try:
        processed_dataset = load_from_disk(processed_path)
    except Exception as e:
        print(f"❌ 错误：加载数据集失败: {e}")
        return

    print("\n" + "="*50)
    print("🔬 开始检查 'train' 数据集的前3条样本...")
    print("="*50)

    # 取出训练集的前3条数据
    samples = processed_dataset["train"].select(range(3))

    for i, sample in enumerate(samples):
        print(f"\n--- 样本 {i+1} ---")
        
        # 将 labels (Token ID 列表) 解码回文本
        # 我们需要过滤掉 -100，因为它不能被解码
        label_ids = [token_id for token_id in sample["labels"] if token_id != -100]
        
        decoded_labels = processor.tokenizer.decode(label_ids, skip_special_tokens=False)
        decoded_labels_skipped = processor.tokenizer.decode(label_ids, skip_special_tokens=True)

        print(f"原始 Label IDs: {sample['labels'][:20]}...") # 只显示前20个ID
        print("-" * 20)
        print(f"解码后的 Labels (包含特殊Token):")
        print(f"➡️  {decoded_labels}")
        print("-" * 20)
        print(f"解码后的 Labels (不含特殊Token):")
        print(f"➡️  {decoded_labels_skipped}")
        
    print("\n" + "="*50)
    print("🕵️‍ 检查结束。")
    print("="*50)


if __name__ == "__main__":
    inspect_data(PROCESSED_DATA_PATH, PROCESSOR_BASE_PATH)