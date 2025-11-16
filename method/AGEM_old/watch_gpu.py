import GPUtil
import time
import subprocess
import os

# --- 配置区 ---
MEMORY_FREE_THRESHOLD = 0.7 # 需要至少 90% 的内存空闲
LOAD_FREE_THRESHOLD = 0.4   # 需要负载低于 10%
CHECK_INTERVAL_SECONDS = 60
COMMAND_TO_RUN = "/data/share/guodong/workspace/code/funetune_whisper/codenew/history/ER_new/finetune_ER_new.py"
# --- 配置区结束 ---

print("GPU 监控启动 (使用 GPUtil)...")
while True:
    try:
        # getAvailable 会返回符合条件的 GPU ID 列表
        # order='first' 表示按ID顺序找第一个可用的
        # maxMemory 和 maxLoad 限制了空闲标准
        available_gpus = GPUtil.getAvailable(
            order='first', 
            limit=1, 
            maxLoad=LOAD_FREE_THRESHOLD, 
            maxMemory=1.0 - MEMORY_FREE_THRESHOLD, # maxMemory 是指最大已用内存比例
            includeNan=False, 
            excludeID=[], 
            excludeUUID=[]
        )
        
        if available_gpus:
            gpu_id = available_gpus[0]
            print(f"\n🎉 发现空闲 GPU: {gpu_id}！准备执行程序...")
            
            # 设置环境变量并执行
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            full_command = COMMAND_TO_RUN
            
            print(f"执行: CUDA_VISIBLE_DEVICES={gpu_id} {full_command}\n")
            subprocess.run(full_command, shell=True, check=True, env=env)
            break # 任务完成
        else:
            print(f"未发现空闲 GPU，将在 {CHECK_INTERVAL_SECONDS} 秒后重试...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except Exception as e:
        print(f"发生错误: {e}", file=sys.stderr)
        time.sleep(CHECK_INTERVAL_SECONDS)