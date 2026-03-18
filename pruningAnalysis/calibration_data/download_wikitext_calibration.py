import os
import json
from datasets import load_dataset

def download_and_clean_wikitext(save_dir, num_samples=1000, words_per_chunk=400):
    """
    下载 Wikitext-2，清洗掉空行和标题，并将文本拼接成固定长度的 chunk。
    words_per_chunk=400 大约对应 500~600 个 Token，足够激活模型的长下文注意力。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"wikitext_calibration_{num_samples}.json")
    
    print("📥 开始从 Hugging Face 下载 Wikitext-2 数据集...")
    # 我们用 train split 来提取大量的校准文本
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    print("🧹 开始清洗无用空行和标题，并拼接上下文...")
    valid_text_blocks = []
    
    # 过滤掉空行和 Wiki 的标题行 (通常以 '=' 开头和结尾)
    for text in dataset['text']:
        cleaned_line = text.strip()
        if cleaned_line and not (cleaned_line.startswith('=') and cleaned_line.endswith('=')):
            valid_text_blocks.append(cleaned_line)
            
    # 将所有干净的段落拼成一个超长字符串
    full_text = " ".join(valid_text_blocks)
    
    # 按词切分
    words = full_text.split()
    
    calibration_texts = []
    # 滑动窗口，将长文本切分成一个个富含上下文的 chunk
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        calibration_texts.append(chunk)
        
        if len(calibration_texts) >= num_samples:
            break

    # 保存到本地磁盘
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(calibration_texts, f, ensure_ascii=False, indent=2)
        
    print("✅ 处理完成！")
    print(f"📁 数据已成功保存至: {save_path}")
    print(f"📊 样本总数: {len(calibration_texts)} 条 (每条约 {words_per_chunk} 个单词)")
    
    # 打印第一条数据看看效果
    print("\n" + "="*50)
    print("预览第 1 条清洗拼接后的长文本:")
    print("="*50)
    print(calibration_texts[0][:500] + "......\n(已截断显示)")
    print("="*50)

if __name__ == "__main__":
    # 你的目标文件夹路径
    TARGET_DIR = "/home/bev/disk4/haoxuan/SpeculativeDecoding/pruningAnalysis/calibration_data"
    
    # 提取 1000 条长文本
    download_and_clean_wikitext(TARGET_DIR, num_samples=1000)