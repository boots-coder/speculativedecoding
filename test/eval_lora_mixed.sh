#!/usr/bin/env bash

set -e

# 简单一键评测脚本：
# 1）使用 LLaMA-Factory 对 LoRA 模型进行 MMLU 通用能力评测
# 2）可选：如果本机安装了 lm-eval-harness，则对 GSM8K 进行推理评测

PROJECT_ROOT="/home/haoxuan/Desktop/SpeculativeDecoding"
LLAMA_FACTORY_DIR="${PROJECT_ROOT}/LLaMA-Factory"

BASE_MODEL="${PROJECT_ROOT}/models/qwen3-4b-passthrough"
LORA_ADAPTER="${PROJECT_ROOT}/models/qwen3-4b-lora-mixed"

# MMLU 评测配置文件
MMLU_EVAL_YAML="${LLAMA_FACTORY_DIR}/qwen3_lora_mmlu_eval.yaml"

echo "==== 1. 使用 LLaMA-Factory 进行 MMLU 评测 ===="
cd "${LLAMA_FACTORY_DIR}"
llamafactory-cli eval "${MMLU_EVAL_YAML}"

echo ""
echo "==== 2.（可选）使用 lm-eval-harness 进行 GSM8K 评测 ===="
echo "如果你已安装 lm-eval-harness，并希望评测 GSM8K，可以手动运行类似命令："
echo ""
echo "  lm_eval \\"
echo "    --model hf \\"
echo "    --model_args pretrained=${BASE_MODEL},peft=${LORA_ADAPTER},dtype=bfloat16 \\"
echo "    --tasks gsm8k \\"
echo "    --batch_size 4 \\"
echo "    --output_path ${PROJECT_ROOT}/models/qwen3-4b-lora-mixed-gsm8k-eval"
echo ""
echo "当前脚本已完成 MMLU 评测，GSM8K 请按需手动调用上面的命令。"

