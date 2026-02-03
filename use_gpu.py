import torch
import time
import os

# ============================
# 配置区
# ============================
# 你要占用的 GPU（0-based index）
TARGET_GPUS = [1, 2, 3]     # 三张卡
MEMORY_FRACTION = 0.85     # 每张卡占用显存比例（95%）
DTYPE = torch.float16      # float16 占用更快
SLEEP_INTERVAL = 10        # 秒

# ============================
# 主逻辑
# ============================
def occupy_gpu(gpu_id: int, fraction: float):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * fraction)

    print(f"[GPU {gpu_id}] Total: {total_mem / 1024**3:.2f} GB")
    print(f"[GPU {gpu_id}] Target: {target_mem / 1024**3:.2f} GB")

    allocated = 0
    tensors = []

    try:
        while allocated < target_mem:
            # 每次申请 256MB
            chunk_size = 256 * 1024 * 1024
            num_elements = chunk_size // torch.tensor([], dtype=DTYPE).element_size()

            t = torch.empty(num_elements, device=device, dtype=DTYPE)
            tensors.append(t)

            allocated += chunk_size
            print(f"[GPU {gpu_id}] Allocated: {allocated / 1024**3:.2f} GB", end="\r")

        print(f"\n[GPU {gpu_id}] Occupation complete.")

        # 保持显存不释放
        while True:
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[GPU {gpu_id}] Releasing memory...")
        del tensors
        torch.cuda.empty_cache()

# ============================
# 入口
# ============================
if __name__ == "__main__":
    assert torch.cuda.device_count() >= 4, "GPU 数量不足 4 张"

    print("Starting GPU memory occupation...")
    for gid in TARGET_GPUS:
        pid = os.fork()
        if pid == 0:
            occupy_gpu(gid, MEMORY_FRACTION)
            exit(0)

    # 父进程等待
    while True:
        time.sleep(60)
