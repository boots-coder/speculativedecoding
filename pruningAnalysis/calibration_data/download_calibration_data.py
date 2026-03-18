import os
import json
from datasets import load_dataset

def download_and_clean_alpaca(save_dir, num_samples=1000):
    # 1. 确保目标文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"alpaca_calibration_{num_samples}.json")
    
    print(f"📥 开始从 Hugging Face 下载 Alpaca 数据集 (提取前 {num_samples} 条)...")
    
    # 2. 加载数据集 (使用 split 直接切片，避免下载/加载不需要的部分)
    # tatsu-lab/alpaca 是最经典的开源 Alpaca 数据集源
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_samples}]")
    
    # 3. 数据清洗与拼接
    print("🧹 开始清洗和拼接文本...")
    calibration_texts = []
    
    for item in dataset:
        # 拼接逻辑：将指令、输入(如果有)和输出组合成一段完整的自然语言上下文
        # 这样能更好地激活模型从浅层到深层的完整认知链路
        text = f"Instruction: {item['instruction']}\n"
        
        # Alpaca 数据有些包含 input，有些没有
        if item.get('input') and item['input'].strip() != "":
            text += f"Input: {item['input']}\n"
            
        text += f"Answer: {item['output']}"
        
        calibration_texts.append(text)

    # 4. 保存到本地磁盘
    with open(save_path, "w", encoding="utf-8") as f:
        # 存为 JSON 列表格式，方便后续的分析脚本直接读取
        json.dump(calibration_texts, f, ensure_ascii=False, indent=2)
        
    print("✅ 处理完成！")
    print(f"📁 数据已成功保存至: {save_path}")
    print(f"📊 样本总数: {len(calibration_texts)}")
    
    # 打印第一条数据看看效果
    print("\n" + "="*40)
    print("预览第 1 条清洗后的数据:")
    print("="*40)
    print(calibration_texts[0])
    print("="*40)

if __name__ == "__main__":
    # 你指定的绝对路径
    TARGET_DIR = "/home/bev/disk4/haoxuan/SpeculativeDecoding/pruningAnalysis/calibration_data"
    
    download_and_clean_alpaca(TARGET_DIR, num_samples=1000)