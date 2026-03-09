---
base_model:
- Qwen/Qwen3-8B
library_name: transformers
tags:
- mergekit
- merge

---
# qwen3-4b-passthrough

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the Passthrough merge method using [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as a base.

### Models Merged

The following models were included in the merge:


### Configuration

The following YAML configuration was used to produce this model:

```yaml
merge_method: passthrough
base_model: Qwen/Qwen3-8B
dtype: bfloat16
tokenizer_source: base

# 这里利用 passthrough + slices 来“深度剪枝”：
# - 保留前 0-20 层
# - 丢弃中间层
# - 保留最后几层（以 Qwen3-8B 当前标准层数 32 为例，保留 28-31 层）
# 如果后续官方确认层数不同，可以按实际层数修改 layer_range。
slices:
  # 前 0-20 层
  - sources:
      - model: Qwen/Qwen3-8B
        layer_range: [0, 18]

  # 最后几层：32-36
  - sources:
      - model: Qwen/Qwen3-8B
        layer_range: [28, 36]



```
