# single_inference.py
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- 1. 请在这里填入你的信息 ---

# 你的微调模型 checkpoint 所在的文件夹路径
MODEL_PATH = "/data/share/guodong/workspace/models/whisper_hf/whisper-small"

# 【请修改】请提供一个你测试集中的音频文件绝对路径 (例如 .mp3, .wav)
AUDIO_FILE_PATH = "/data/share/guodong/workspace/datasets/CommonVoice/zh-CN/clips/common_voice_zh-CN_22148240.mp3" 

# 【请修改】请在这里写下上面这个音频文件对应的【正确】文字内容
CORRECT_TRANSCRIPTION = "但是书店没货"

# --- 结束填写 ---


def main():
    print("="*80)
    print("🔬 开始单样本推理诊断...")
    print("="*80)

    # 2. 加载模型和处理器
    print(f"🔍 正在从 '{MODEL_PATH}' 加载模型和处理器...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
        model.to("cuda") # 将模型移动到GPU
        model.eval()     # 设置为评估模式
        print("✅ 模型和处理器加载成功。")
    except Exception as e:
        print(f"❌ 致命错误：加载模型失败！请检查 MODEL_PATH 是否正确。")
        print(f"   错误详情: {e}")
        return

    # 3. 加载并处理单个音频文件
    print(f"\n🎧 正在加载音频文件: {AUDIO_FILE_PATH}")
    try:
        # 使用 librosa 加载音频，并确保采样率为16kHz
        audio_array, sampling_rate = librosa.load(AUDIO_FILE_PATH, sr=16000)
        # 使用 feature_extractor 处理音频
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        print("✅ 音频处理成功。")
    except Exception as e:
        print(f"❌ 致命错误：加载或处理音频失败！请检查 AUDIO_FILE_PATH 是否正确。")
        print(f"   错误详情: {e}")
        return

    # 4. 使用两种不同策略进行解码
    print("\n" + "-"*80)
    print("🚀 开始解码...")
    print("-"*80)

    # --- 方法一：强制指定中文进行解码 ---
    # 这是我们期望它能正确工作的方式
    print("\n--- [方法一] 强制指定中文进行解码 ---")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    predicted_ids_forced = model.generate(input_features.to("cuda"), forced_decoder_ids=forced_decoder_ids)
    transcription_forced = processor.batch_decode(predicted_ids_forced, skip_special_tokens=True)[0]
    print(f"模型输出:  {transcription_forced}")

    # --- 方法二：让模型自动检测语言进行解码 ---
    # 这是你上次测试失败的方式
    print("\n--- [方法二] 让模型自动检测语言进行解码 ---")
    predicted_ids_auto = model.generate(input_features.to("cuda"))
    transcription_auto = processor.batch_decode(predicted_ids_auto, skip_special_tokens=True)[0]
    print(f"模型输出:  {transcription_auto}")

    # 5. 结果对比
    print("\n" + "="*80)
    print("📊 最终结果对比")
    print("="*80)
    print(f"✅ 正确文本:      {CORRECT_TRANSCRIPTION}")
    print(f"🤔 方法一输出:    {transcription_forced}")
    print(f"🤔 方法二输出:    {transcription_auto}")
    print("="*80)


if __name__ == "__main__":
    # 检查路径是否已被修改
    if "/path/to/your/test_audio.mp3" in AUDIO_FILE_PATH:
        print("❌ 错误：请先修改脚本中的 AUDIO_FILE_PATH 和 CORRECT_TRANSCRIPTION 变量！")
    else:
        main()