import sys
from pathlib import Path
from typing import List

# 保证从任意工作目录运行都能正确导入 config 和 test 下模块
_root = Path(__file__).resolve().parent.parent
_test_dir = _root / "test"
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))

from math_eval_data import MATH_EVAL_ITEMS, check_math_answer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config.config import PRUNED_MODEL_PATH, DEVICE, HF_TOKEN

# 数学 LoRA：qwen3-4b-math-lora（底座为 qwen3-4b-passthrough）
LORA_MATH_PATH = "/home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-math-lora"


MATH_SYSTEM = "你是一个数学助手。请清晰写出推理与计算步骤，给出准确答案。"
COMMON_SENSE_SYSTEM = "你是一个乐于助人、逻辑清晰的AI助手。请用简明扼要的语言回答问题。"


def get_common_sense_questions() -> List[str]:
  """与 test_lora_inference.py 中一致的常识题。"""
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


def load_lora_model_and_tokenizer():
  root = Path(__file__).resolve().parent.parent
  model_path = (root / PRUNED_MODEL_PATH).as_posix()

  print(f"Loading base pruned model from: {model_path}")
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    token=HF_TOKEN,
    trust_remote_code=True,
  )
  
  base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    trust_remote_code=True,
  )
  
  print(f"Loading LoRA weights from: {LORA_MATH_PATH}")
  model = PeftModel.from_pretrained(base_model, LORA_MATH_PATH)
  model = model.merge_and_unload()
  print("LoRA (math) weights merged successfully!")
  
  return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, system_prompt: str = MATH_SYSTEM) -> str:
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": question}
  ]
  
  # 2. 移除手工拼装，完全交给官方 tokenizer 处理模板
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  
  inputs = tokenizer([text], return_tensors="pt").to(model.device)
  
  

  # 3. 结合官方推荐参数 + 剪枝模型保护参数
  output_ids = model.generate(
    **inputs,
    max_new_tokens=256,       
    do_sample=True,           
    temperature=0.7,          # 官方 Instruct 推荐值
    top_p=0.8,                # 官方 Instruct 推荐值
    top_k=20,                 # 官方 Instruct 推荐值
  )
  
  input_len = inputs["input_ids"].shape[1]
  new_output_ids = output_ids[0][input_len:]
  text = tokenizer.decode(new_output_ids, skip_special_tokens=True)
  
  return text.strip()


def main() -> None:
  root = Path(__file__).resolve().parent.parent
  output_dir = root / "output"
  output_dir.mkdir(parents=True, exist_ok=True)

  common_sense_questions = get_common_sense_questions()

  model, tokenizer = load_lora_model_and_tokenizer()
  out_path = output_dir / "qwen3_4b_math_lora_answers.txt"

  with out_path.open("w", encoding="utf-8") as fout:
    # 1. 数学题（仅 10 道，带准确率评估）
    fout.write("========== 数学题（10 道，计准确率）==========\n\n")
    eval_results: List[bool] = []
    for idx, (q, expected_list) in enumerate(MATH_EVAL_ITEMS, 1):
      print(f"\n=== [数学] Q{idx}: {q}")
      ans = generate_answer(model, tokenizer, q, system_prompt=MATH_SYSTEM)
      ok = check_math_answer(ans, expected_list)
      eval_results.append(ok)
      print("[Math LoRA]", ans[:200].replace("\n", " "))
      fout.write(f"Q{idx}: {q}\nA: {ans}\n正确: {ok}\n\n")
    correct, total = sum(eval_results), len(eval_results)
    accuracy = correct / total if total else 0.0
    eval_summary = f">>> 数学题准确率: {correct}/{total} = {accuracy:.1%}\n\n"
    fout.write(eval_summary)
    print(eval_summary.strip())

    # 2. 常识题（与 test_lora_inference 一致）
    fout.write("========== 常识题 ==========\n\n")
    for idx, q in enumerate(common_sense_questions, 1):
      print(f"\n=== [常识] Q{idx}: {q}")
      ans = generate_answer(model, tokenizer, q, system_prompt=COMMON_SENSE_SYSTEM)
      print("[Math LoRA]", ans[:200].replace("\n", " "))
      fout.write(f"Q{idx}: {q}\nA: {ans}\n\n")

  print(f"\nDone. Math LoRA answers (数学+常识) saved to: {out_path}")


if __name__ == "__main__":
  main()