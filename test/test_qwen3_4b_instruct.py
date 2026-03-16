import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 修改为你刚刚下载的具体路径
MODEL_PATH = "/home/haoxuan/Desktop/SpeculativeDecoding/models/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 test_lora_math 相同的数学评估题
_root = Path(__file__).resolve().parent.parent
_test_dir = _root / "test"
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from math_eval_data import MATH_EVAL_ITEMS, check_math_answer

MATH_SYSTEM = "你是一个数学助手。请清晰写出推理与计算步骤，给出准确答案。"
DEFAULT_SYSTEM = "你是一个乐于助人、逻辑清晰的AI助手。请用简明扼要的语言回答问题。"

def get_questions() -> List[str]:
    # 保持一致的 10 个问题
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
    print(f"Loading instruct model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto", # 使用模型默认的 dtype (通常是 bfloat16)
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": question}
  ]


  
    # 2. 调用 apply_chat_template 
    # 注：Qwen3-4B-Instruct-2507 不支持思考模式，无需传 enable_thinking
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 3. 使用官方 Model Card 推荐的非思考模式采样参数
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,     # 官方建议 Instruct 模型日常 16K，这里设 256 足够 10 个问题
        do_sample=False,          
        temperature=0.1,         # 官方推荐
        top_p=1,               # 官方推荐
        top_k=1,                # 官方推荐
    )
    
    # 4. 截断输入，只保留新生成的内容
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 5. 解码
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return content


def main() -> None:
    # 获取当前脚本所在目录的父目录作为 root
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = get_questions()
    model, tokenizer = load_model_and_tokenizer()
    out_path = output_dir / "qwen3_4b_instruct_answers.txt"

    with out_path.open("w", encoding="utf-8") as fout:
        # 1. 数学题评估（10 道，与 test_lora_math 相同）
        fout.write("========== 数学题评估（10 道，计准确率）==========\n\n")
        eval_results = []
        for idx, (q, expected_list) in enumerate(MATH_EVAL_ITEMS, 1):
            print(f"\n=== [数学评估] Q{idx}: {q}")
            ans = generate_answer(model, tokenizer, q, system_prompt=MATH_SYSTEM)
            ok = check_math_answer(ans, expected_list)
            eval_results.append(ok)
            print(f"[4B-Instruct] {ans[:200].replace(chr(10), ' ')}...")
            fout.write(f"Q{idx}: {q}\nA: {ans}\n正确: {ok}\n\n")
        correct, total = sum(eval_results), len(eval_results)
        accuracy = correct / total if total else 0.0
        eval_summary = f">>> 数学题评估准确率: {correct}/{total} = {accuracy:.1%}\n\n"
        fout.write(eval_summary)
        print(eval_summary.strip())

        # 2. 常识题（原 10 题）
        fout.write("========== 常识题 ==========\n\n")
        for idx, q in enumerate(questions, 1):
            print(f"\n=== Q{idx}: {q} ===")
            ans = generate_answer(model, tokenizer, q)
            print(f"[4B-Instruct] {ans[:200].replace(chr(10), ' ')}...")
            fout.write(f"Q{idx}: {q}\nA(4B-Instruct): {ans}\n\n{'-'*50}\n\n")

    print(f"\nDone. Instruct answers saved to: {out_path}")


if __name__ == "__main__":
    main()