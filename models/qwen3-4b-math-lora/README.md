---
library_name: peft
license: other
base_model: /home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough
tags:
- base_model:adapter:/home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3-4b-math-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-4b-math-lora

This model is a fine-tuned version of [/home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough](https://huggingface.co//home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough) on the open_r1_math dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7475

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.7941        | 0.8421 | 500  | 0.8012          |
| 0.7575        | 1.6838 | 1000 | 0.7591          |
| 0.7313        | 2.5255 | 1500 | 0.7480          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2