# load_commonvoice.py (修改后的版本)

from datasets import load_dataset, DatasetDict

# --- 关键修改 ---
# 1. 指向你刚刚在项目里保存的本地加载脚本
LOCAL_SCRIPT_PATH = "/data/share/guodong/workspace/code/funetune_whisper/codenew/history/load_cv/common_voice_22_0.py"  # 假设它和你的运行脚本在同一个目录

# 2. 指向你下载好的 Common Voice 文件夹的 **上一级** 目录
#    加载脚本会自动根据下面的 name='zh-CN' 来寻找里面的 zh-CN 文件夹
LOCAL_DATA_DIR = "/data/share/guodong/workspace/datasets/CommonVoice"


# 创建一个空的数据集字典
common_voice = DatasetDict()

print(f"✅ 正在从本地脚本 '{LOCAL_SCRIPT_PATH}' 加载数据集...")

# 3. 调用 load_dataset，第一个参数是本地脚本路径，并使用 name 参数指定语言
common_voice["train"] = load_dataset(
    LOCAL_SCRIPT_PATH, 
    name="zh-CN",        # 使用 name 参数指定语言
    split="train+validation", 
    data_dir=LOCAL_DATA_DIR,
    trust_remote_code=True # <-- 对于新版datasets，需要这个参数来允许执行本地脚本
)

common_voice["test"] = load_dataset(
    LOCAL_SCRIPT_PATH, 
    name="zh-CN",
    split="test", 
    data_dir=LOCAL_DATA_DIR,
    trust_remote_code=True
)

print("✅ 成功从本地路径和本地脚本加载数据集！")
print(common_voice)