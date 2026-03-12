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
- name: pruned_qwen_recovered
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pruned_qwen_recovered

This model is a fine-tuned version of [/home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough](https://huggingface.co//home/haoxuan/Desktop/SpeculativeDecoding/models/qwen3-4b-passthrough) on the alpaca_zh and the alpaca_en datasets.
It achieves the following results on the evaluation set:
- Loss: 1.5954

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
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

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 1.9241        | 0.0819 | 500   | 1.9055          |
| 1.8380        | 0.1637 | 1000  | 1.8345          |
| 1.8106        | 0.2456 | 1500  | 1.7891          |
| 1.8127        | 0.3275 | 2000  | 1.7663          |
| 1.6428        | 0.4094 | 2500  | 1.7416          |
| 1.6549        | 0.4912 | 3000  | 1.7202          |
| 1.7719        | 0.5731 | 3500  | 1.7020          |
| 1.7239        | 0.6550 | 4000  | 1.6821          |
| 1.8112        | 0.7369 | 4500  | 1.6662          |
| 1.6038        | 0.8187 | 5000  | 1.6486          |
| 1.5904        | 0.9006 | 5500  | 1.6397          |
| 1.7034        | 0.9825 | 6000  | 1.6246          |
| 1.1983        | 1.0644 | 6500  | 1.6625          |
| 1.3806        | 1.1462 | 7000  | 1.6603          |
| 1.1792        | 1.2281 | 7500  | 1.6556          |
| 1.4217        | 1.3100 | 8000  | 1.6320          |
| 1.1326        | 1.3918 | 8500  | 1.6263          |
| 1.1700        | 1.4737 | 9000  | 1.6237          |
| 1.1848        | 1.5556 | 9500  | 1.6161          |
| 1.1782        | 1.6375 | 10000 | 1.6033          |
| 1.1903        | 1.7193 | 10500 | 1.5935          |
| 1.1837        | 1.8012 | 11000 | 1.5843          |
| 1.2887        | 1.8831 | 11500 | 1.5775          |
| 1.2612        | 1.9650 | 12000 | 1.5706          |
| 0.7587        | 2.0468 | 12500 | 1.5992          |
| 0.6810        | 2.1287 | 13000 | 1.6044          |
| 0.7173        | 2.2106 | 13500 | 1.5939          |
| 0.6897        | 2.2925 | 14000 | 1.5990          |
| 0.7292        | 2.3743 | 14500 | 1.5941          |
| 0.6923        | 2.4562 | 15000 | 1.5974          |
| 0.6528        | 2.5381 | 15500 | 1.6013          |
| 0.5610        | 2.6200 | 16000 | 1.6014          |
| 0.6665        | 2.7018 | 16500 | 1.6019          |
| 0.6615        | 2.7837 | 17000 | 1.6007          |
| 0.7075        | 2.8656 | 17500 | 1.5949          |
| 0.6569        | 2.9474 | 18000 | 1.5954          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2