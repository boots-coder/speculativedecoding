#!/usr/bin/env bash
# 串行跑通：官方 4B、初步 LoRA 4B、数学 LoRA 4B，并输出对比表格
# 常识回答：见 output/ 下各文件「常识题」部分肉眼对比
# 数学能力：从各文件解析准确率并制表

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUTPUT_DIR="$ROOT/output"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  1/3  官方 4B (Qwen3-4B-Instruct)"
echo "=============================================="
python test/test_qwen3_4b_instruct.py

echo ""
echo "=============================================="
echo "  2/3  初步 LoRA 4B (pruned + recovered)"
echo "=============================================="
python test/test_lora_inference.py

echo ""
echo "=============================================="
echo "  3/3  数学 LoRA 4B (qwen3-4b-math-lora)"
echo "=============================================="
python test/test_lora_math.py

# 从 output 文件里提取数学准确率（兼容 "数学题准确率" 与 "数学题评估准确率"）
get_accuracy() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "N/A"
    return
  fi
  local line
  line=$(grep -E ">>> 数学题(评估)?准确率:" "$f" 2>/dev/null | head -1)
  if [[ -z "$line" ]]; then
    echo "N/A"
    return
  fi
  # 去掉 ">>> 数学题(评估)准确率: " 前缀，得到 "8/10 = 80.0%"
  echo "$line" | sed 's/.*: *//' | tr -d '\r'
}

ACC_INSTRUCT=$(get_accuracy "$OUTPUT_DIR/qwen3_4b_instruct_answers.txt")
ACC_LORA=$(get_accuracy "$OUTPUT_DIR/qwen3_4b_lora_recovered_answers.txt")
ACC_MATH=$(get_accuracy "$OUTPUT_DIR/qwen3_4b_math_lora_answers.txt")

echo ""
echo "=============================================="
echo "  对比表格"
echo "=============================================="
echo ""
printf "%-20s | %-45s | %s\n" "模型" "常识回答（肉眼观察 output 文件）" "数学能力 (accuracy)"
printf "%s\n" "---------------------|-----------------------------------------------|------------------"
printf "%-20s | %-45s | %s\n" "官方 4B" "output/qwen3_4b_instruct_answers.txt 常识题" "${ACC_INSTRUCT:-N/A}"
printf "%-20s | %-45s | %s\n" "初步 LoRA 4B" "output/qwen3_4b_lora_recovered_answers.txt 常识题" "${ACC_LORA:-N/A}"
printf "%-20s | %-45s | %s\n" "数学 LoRA 4B" "output/qwen3_4b_math_lora_answers.txt 常识题" "${ACC_MATH:-N/A}"
echo ""
echo "常识回答：请直接打开上述 output 文件中「常识题」部分对比。"
echo "数学能力：上表为 10 道数学题的准确率 (正确数/10 = 百分比)。"
echo ""
