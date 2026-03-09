from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import PRUNED_MODEL_PATH, DEVICE, HF_TOKEN


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


def load_pruned_model_and_tokenizer():
  root = Path(__file__).resolve().parent.parent
  model_path = (root / PRUNED_MODEL_PATH).as_posix()

  print(f"Loading pruned model from: {model_path}")
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    token=HF_TOKEN,
    trust_remote_code=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,  # 保持与你原脚本一致
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    trust_remote_code=True,
  )
  return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str) -> str:
  # 1. 采用官方推荐的 messages 列表结构
  messages = [
      {"role": "user", "content": question}
  ]
  
  # 2. 完全交给官方 tokenizer 处理模板
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  
  inputs = tokenizer([text], return_tensors="pt").to(model.device)
  
  # 提取 Qwen 的结束符
  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|im_end|>")
  ]
  terminators = [t for t in terminators if t is not None]

  # 3. 结合官方推荐参数 + 剪枝模型保护参数
  output_ids = model.generate(
    **inputs,
    max_new_tokens=256,       
    do_sample=True,           # 必须开启采样，避免基座模型死锁
    temperature=0.7,          
    top_p=0.8,                
    top_k=20,                 
    eos_token_id=terminators, 
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
  )
  
  # 4. 使用 token 长度截断
  input_len = inputs["input_ids"].shape[1]
  new_output_ids = output_ids[0][input_len:]
  text = tokenizer.decode(new_output_ids, skip_special_tokens=True)
  
  return text.strip()


def main() -> None:
  root = Path(__file__).resolve().parent.parent
  output_dir = root / "output"
  output_dir.mkdir(parents=True, exist_ok=True)

  questions = get_questions()

  model, tokenizer = load_pruned_model_and_tokenizer()

  out_path = output_dir / "qwen3_4b_pruned_only_answers.txt"

  with out_path.open("w", encoding="utf-8") as fout:
    for idx, q in enumerate(questions, 1):
      print(f"\n=== Q{idx}: {q}")
      ans = generate_answer(model, tokenizer, q)
      print("[4B-Pruned]", ans[:200].replace("\n", " "))
      fout.write(f"Q{idx}: {q}\nA(4B-Pruned): {ans}\n\n")

  print(f"\nDone. Pruned-only answers saved to: {out_path}")


if __name__ == "__main__":
  main()