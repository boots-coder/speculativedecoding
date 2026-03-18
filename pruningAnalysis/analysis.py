import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def calculate_layer_redundancy(model_path, calibration_texts, device="cuda"):
    print(f"Loading model and tokenizer from {model_path}...")
    # 建议使用 bfloat16 加载以节省显存，并保持与训练一致的精度
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} hidden layers. Starting similarity analysis...\n")

    # 用于累加每一层的余弦相似度
    layer_similarities = {i: 0.0 for i in range(1, num_layers + 1)}
    
    with torch.no_grad():
        for text in tqdm(calibration_texts, desc="Processing Calibration Data"):
            # 对输入进行 Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            
            # 开启 output_hidden_states 以获取每一层的特征
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # 遍历每一层计算相似度
            for i in range(1, num_layers + 1):
                # in_state: 第 i 层的输入 (shape: [batch, seq_len, hidden_dim])
                # out_state: 第 i 层的输出
                in_state = hidden_states[i-1]
                out_state = hidden_states[i]
                
                # 在 hidden_dim 维度 (dim=-1) 上计算余弦相似度，然后对 batch 和 seq_len 取平均
                sim = F.cosine_similarity(in_state, out_state, dim=-1).mean().item()
                layer_similarities[i] += sim

    # 计算平均相似度
    num_samples = len(calibration_texts)
    for i in range(1, num_layers + 1):
        layer_similarities[i] /= num_samples

    # 按照相似度从高到低排序 (相似度越高，说明该层做的变换越少，越冗余)
    sorted_layers = sorted(layer_similarities.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    print("🎯 层冗余度排行榜 (Similarity 越接近1，越建议剪除)")
    print("="*50)
    for rank, (layer_idx, sim) in enumerate(sorted_layers):
        print(f"Rank {rank+1:02d} | Layer {layer_idx:02d} | Cosine Similarity: {sim:.4f}")

    # 选出最冗余的 18 层
    layers_to_prune = sorted([layer_idx for layer_idx, _ in sorted_layers[:18]])
    layers_to_keep = sorted([layer_idx for layer_idx, _ in sorted_layers[18:]])
    
    print("\n" + "="*50)
    print("✂️  推荐剪枝方案 (Top 18 最冗余层)")
    print("="*50)
    print(f"建议剪除的层 (Prune): {layers_to_prune}")
    print(f"建议保留的层 (Keep) : {layers_to_keep}")

    return layers_to_keep, layers_to_prune

if __name__ == "__main__":
    import json
    
    MODEL_PATH = "Qwen/Qwen3-8B" # 模型路径
    DATA_PATH = "/home/bev/disk4/haoxuan/SpeculativeDecoding/pruningAnalysis/calibration_data/alpaca_calibration_1000.json"
    
    # 从刚刚保存的 JSON 文件加载校准数据
    print(f"Loading calibration data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        calibration_data = json.load(f)
    
    # 执行探测 (由于1000条加上36层输出会比较慢，你可以先切片 calibration_data[:100] 测试一下速度)
    keep_layers, prune_layers = calculate_layer_redundancy(MODEL_PATH, calibration_data[:100])