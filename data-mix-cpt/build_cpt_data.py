import os
# ==========================================
# 0. 终极防爆盘设置：把 Hugging Face 缓存强行指到 disk4
# ==========================================
os.environ["HF_HOME"] = "/home/bev/disk4/haoxuan/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/home/bev/disk4/haoxuan/hf_cache/datasets"

import multiprocessing
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# ==========================================
# 1. 核心参数配置
# ==========================================
MODEL_PATH = "Qwen/Qwen3-8B" 
OUTPUT_DIR = "/home/bev/disk4/haoxuan/SpeculativeDecoding/data-mix-cpt-full"
SEQ_LENGTH = 4096             
NUM_PROC = multiprocessing.cpu_count() 

print(f"🚀 启动 CPT 全量数据构建流水线 | CPU 核心数: {NUM_PROC} | 目标长度: {SEQ_LENGTH}")

# ==========================================
# 2. 加载 Tokenizer
# ==========================================
print("📦 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
eos_token_id = tokenizer.eos_token_id 

# ==========================================
# 3. 全量拉取数据集 (支持断点续传)
# ==========================================
print("📥 开始硬核全量拉取数据 (若遇网络中断，直接重新运行脚本即可断点续传)...")

# 你可以根据实际需要的量级修改这里的切片大小，比如加个 0
ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:50000]")
ds_math = load_dataset("open-web-math/open-web-math", split="train[:20000]")


# 替换为这行新代码 (不用加 trust_remote_code)
ds_zh = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[:15000]")
# 替换为纯 Parquet 格式的 SlimPajama，秒级解析！
ds_pajama = load_dataset("DKYoon/SlimPajama-6B", split="train[:15000]")
# ------------------------------------------
# 统一格式：提取 "text" 列
# ------------------------------------------
def keep_only_text(example, text_col="text"):
    return {"text": example[text_col]}

print("🧹 统一数据格式并清理冗余列...")
ds_fineweb = ds_fineweb.map(keep_only_text, remove_columns=ds_fineweb.column_names, num_proc=NUM_PROC)
ds_math = ds_math.map(keep_only_text, remove_columns=ds_math.column_names, num_proc=NUM_PROC)
ds_zh = ds_zh.map(keep_only_text, remove_columns=ds_zh.column_names, num_proc=NUM_PROC)
ds_pajama = ds_pajama.map(keep_only_text, remove_columns=ds_pajama.column_names, num_proc=NUM_PROC)

# 全局洗牌
print("🔀 拼接数据集并进行全局洗牌 (Global Shuffle)...")
mixed_dataset = concatenate_datasets([ds_fineweb, ds_math, ds_zh, ds_pajama])
mixed_dataset = mixed_dataset.shuffle(seed=42)
print(f"✅ 混合完成！当前总样本数: {len(mixed_dataset)}")

# ==========================================
# 4. 高速 Tokenization
# ==========================================
print("🔠 开始执行多进程 Tokenization...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False, padding=False)

tokenized_datasets = mixed_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"], 
    desc="Tokenizing",
)
# ==========================================
# 5. 序列打包 (Sequence Packing)
# ==========================================
print("📦 开始执行序列打包 (Sequence Packing)...")
def group_texts(examples):
    concatenated_ids = []
    for ids in examples["input_ids"]:
        concatenated_ids.extend(ids)
        concatenated_ids.append(eos_token_id) 

    total_length = len(concatenated_ids)
    if total_length >= SEQ_LENGTH:
        total_length = (total_length // SEQ_LENGTH) * SEQ_LENGTH
        
    result = {
        "input_ids": [
            concatenated_ids[i : i + SEQ_LENGTH]
            for i in range(0, total_length, SEQ_LENGTH)
        ],
        "labels": [
            concatenated_ids[i : i + SEQ_LENGTH]
            for i in range(0, total_length, SEQ_LENGTH)
        ],
        # Add a dense attention mask since our sequences are fully packed
        "attention_mask": [
            [1] * SEQ_LENGTH
            for _ in range(0, total_length, SEQ_LENGTH)
        ]
    }
    return result

packed_dataset = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
    remove_columns=tokenized_datasets.column_names, # 🚨 CRITICAL: Drop the old mismatched columns!
    desc=f"Packing into blocks of {SEQ_LENGTH}",
)

print(f"🎯 打包完成！最终得到 {len(packed_dataset)} 个绝对等长 ({SEQ_LENGTH}) 的训练块。")

# ==========================================
# 6. 保存到磁盘
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"💾 正在将处理好的数据集保存为极速加载格式至: {OUTPUT_DIR} ...")
packed_dataset.save_to_disk(OUTPUT_DIR)

print("\n🎉 全部处理完毕！纯正的流食大餐已备好！")