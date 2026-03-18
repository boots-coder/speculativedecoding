import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import os

# ================= 路径配置 =================
TEACHER_MODEL_PATH = "Qwen/Qwen3-8B" # 如果有本地路径请替换
STUDENT_MODEL_PATH = "/home/bev/disk4/haoxuan/SpeculativeDecoding/model/qwen3-4b-pruned-data-driven"
DATA_PATH = "/home/bev/disk4/haoxuan/SpeculativeDecoding/data-mix-cpt-full"
OUTPUT_DIR = "/home/bev/disk4/haoxuan/SpeculativeDecoding/recoverTraining/checkpoints"

# ================= 自定义 KD Trainer =================
class KDCptTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        
        # 确保 Teacher 不参与参数更新
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # 1. Student 前向传播 (需要计算梯度)
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # 2. Teacher 前向传播 (不计算梯度)
        with torch.no_grad():
            # 将 Teacher 动态移至与 Student 相同的设备上
            if self.teacher.device != student_logits.device:
                self.teacher.to(student_logits.device)
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # 3. 自回归偏移 (Shift)
        # Token N 预测 Token N+1，所以 Logits 去掉最后一个时间步
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

        # 展平以便计算 KL 散度
        vocab_size = shift_student_logits.size(-1)
        shift_student_logits = shift_student_logits.view(-1, vocab_size)
        shift_teacher_logits = shift_teacher_logits.view(-1, vocab_size)

        # 4. 计算纯 KL 散度损失
        # PyTorch 的 kl_div 要求: input 是 log_softmax, target 是 softmax 且是普通的概率空间
        student_log_probs = F.log_softmax(shift_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        # batchmean 会对 batch 维度求平均
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        
        # 乘以 T^2 补偿梯度缩放
        loss = kl_loss * (self.temperature ** 2)

        return (loss, student_outputs) if return_outputs else loss

# ================= 主函数 =================
def main():
    # 1. 加载数据
    print("Loading packed dataset...")
    dataset = load_from_disk(DATA_PATH)
    
    # 2. 加载模型
    # Teacher 可以用 fp16/bf16 加载以节省显存
    print("Loading Teacher Model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto" # 自动分配显存
    )
    
    print("Loading Student Model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 3. 训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,   # 根据显存调整，4096 长度非常吃显存
        gradient_accumulation_steps=8,   # 累积梯度，模拟大 Batch Size
        learning_rate=2e-5,              # CPT 学习率通常较小
        bf16=True,                       # 强烈建议开启 bf16
        logging_steps=10,
        save_steps=500,
        max_steps=5000,                  # 或者用 num_train_epochs
        gradient_checkpointing=True,     # 开启梯度检查点，极致省显存
        report_to="tensorboard",
        remove_unused_columns=False      # 关键: 防止 Trainer 自动删掉你的 input_ids
    )

    # 4. 初始化自定义 Trainer
    trainer = KDCptTrainer(
        teacher_model=teacher_model,
        temperature=2.0,                 # 温度系数，通常在 1.0 到 4.0 之间
        model=student_model,
        args=training_args,
        train_dataset=dataset,
    )

    # 5. 开始训练
    print("Starting Distillation CPT...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_student_model"))
    print("Training Complete!")

if __name__ == "__main__":
    main()