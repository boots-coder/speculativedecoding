import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import BASE_MODEL_ID, DEVICE, HF_TOKEN

def get_questions() -> List[str]:
    # 跟对比脚本保持一致的 10 个问题
    return [
        "用一句话解释什么是量子纠缠？",
        "为什么天空是蓝色的？",
        "写一段 50 字左右的自我介绍。",
        "给出三个提高深度学习训练稳定性的建议。",
        "设计一个每天早晨 10 分钟的高效学习计划。",
        "Explain the difference between supervised and unsupervised learning in one paragraph.",
        "Give me a simple Python function that computes Fibonacci numbers iteratively.",
        "假设你是一个产品经理，如何描述一款面向大学生的学习助手应用？",
        "请用通俗的语言解释大 O 时间复杂度。",
        "你认为未来五年大语言模型最重要的应用场景是什么？",
    ]

def load_model_and_tokenizer():
    print(f"Loading model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    return model, tokenizer

@torch.inference_mode()
def generate_answer(model, tokenizer, question: str) -> str:
    # 1. 使用官方推荐的 messages 格式
    messages = [
        {"role": "system", "content": "你是一个乐于助人、逻辑清晰的AI助手。请用简明扼要的语言回答问题。"},
        {"role": "user", "content": question}
    ]
    
    # 2. 调用 apply_chat_template (支持 Qwen3 的特殊 tag)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # 默认关闭思考模式
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 3. 使用官方推荐的思考模式采样参数 (严格禁止 do_sample=False)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,     # 如果有思考过程，则 Token 必须给够
        do_sample=True,          
        temperature=0.6,         # 官方推荐值
        top_p=0.95,              # 官方推荐值
        top_k=20                 # 官方推荐值
    )
    
    # 4. 截取新生成的 Token ID
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 5. 官方提供的解析逻辑：分离 <think> 内容和最终回答
    try:
        # 151668 是 Qwen3 中 </think> 的 token ID
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # 将思考过程和最终回答拼接在一起返回，方便查看它的推理逻辑
    if thinking_content:
        final_text = f"【思考过程】\n{thinking_content}\n\n【最终回答】\n{content}"
    else:
        final_text = content
        
    return final_text

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = get_questions()
    model, tokenizer = load_model_and_tokenizer()

    out_path = output_dir / "qwen3_8b_instruct_answers.txt"

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, q in enumerate(questions, 1):
            print(f"\n=== Q{idx}: {q} ===")
            ans = generate_answer(model, tokenizer, q)
            # 打印前 200 个字符预览
            print(ans[:200].replace("\n", " ") + "...") 
            fout.write(f"Q{idx}: {q}\nA(8B):\n{ans}\n\n{'-'*50}\n\n")

    print(f"\nDone. Official format answers saved to: {out_path}")

if __name__ == "__main__":
    main()