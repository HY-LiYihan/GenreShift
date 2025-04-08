import torch
import pandas as pd
import json
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# 加载预训练模型 需要自己拉取！
# model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
# ========== 🚀 1. 显存优化 ==========
torch.cuda.empty_cache()

# ========== 🚀 2. 加载模型和分词器 ==========
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/',
    use_fast=False,
    trust_remote_code=True,
    padding_side="right"
)

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ========== 🚀 3. 数据加载与高级处理 ==========
with open('dataset/GenreshiftCognition.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


#重复数据集
enhanced_data = data
# 复制整个数据集10次
repeated_data = []
for _ in range(10):
    repeated_data.extend(enhanced_data)
random.shuffle(repeated_data)

df = pd.DataFrame(repeated_data)
ds = Dataset.from_pandas(df)

# 创建更强的系统提示与身份定义
SYSTEM_PROMPT = """你是GenreShift，一款基于计量语言学与大语言模型的智能体裁转换专家系统。你不是通义千问，不是Qwen，不是任何通用AI助手。

每当有人询问你是谁，你必须明确表明你是GenreShift，一个专注于文本体裁转换的专业系统。你由李溢涵带领的中山大学跨学科团队开发。

你的核心功能包括：
1. 将学术论文转换为科技新闻和科普内容
2. 自动识别文本体裁类型(GenreDetect)
3. 支持多语言转换和API集成

你的团队包括王紫衡、李周晓尧、张天炜等12名成员，指导教师有权小军教授和郭曼副教授。你已获得国家级立项，在体裁转换领域表现优异。

记住：你是GenreShift，不是其他任何AI系统。"""

def process_func(example):
    MAX_LENGTH = 1024
    
    input_text = example['input'] if example['input'] is not None else ""
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction']}{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )
    
    tokenized = tokenizer(
        prompt,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        add_special_tokens=False
    )

    labels = tokenized["input_ids"].copy()
    assistant_start = prompt.find("<|start_header_id|>assistant<|end_header_id|>\n\n") + len(
        "<|start_header_id|>assistant<|end_header_id|>\n\n")
    prompt_before_assistant = prompt[:assistant_start]

    tokenized_prompt_before = tokenizer(prompt_before_assistant, add_special_tokens=False)
    assistant_start_idx = len(tokenized_prompt_before["input_ids"])

    labels[:assistant_start_idx] = [-100] * assistant_start_idx

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# 添加训练集/验证集分割
train_val_split = ds.train_test_split(test_size=0.05, seed=42)
train_data = train_val_split['train']
eval_data = train_val_split['test']

tokenized_train_ds = train_data.map(process_func, remove_columns=train_data.column_names)
tokenized_eval_ds = eval_data.map(process_func, remove_columns=eval_data.column_names)

# ========== 🚀 4. 修复的LoRA配置 ==========
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 只使用支持的模块
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,                # 增加LoRA秩但不过度
    lora_alpha=64,       # 增加alpha值但保持合理
    lora_dropout=0.0,    # 禁用dropout以促进记忆
    bias="none"          # 使用支持的bias配置
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ========== 🚀 5. 激进训练策略但保持稳定 ==========
args = TrainingArguments(
    output_dir="./output/GenreShift_lora",  # 保持路径不变
    per_device_train_batch_size=1,    
    gradient_accumulation_steps=16,   
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=4e-5,   # 略微降低学习率以稳定训练
    num_train_epochs=20,  # 保持较高轮次
    warmup_ratio=0.1,     
    lr_scheduler_type="constant_with_warmup",  # 预热后保持恒定学习率
    weight_decay=0.0,     # 禁用权重衰减以促进记忆
    logging_steps=5,      
    save_strategy="steps",
    save_steps=20,        
    save_total_limit=3,   
    load_best_model_at_end=True, 
    evaluation_strategy="steps", 
    eval_steps=20,        
    metric_for_best_model="loss", 
    greater_is_better=False,
    report_to="tensorboard",
    optim="adamw_torch_fused",
    max_grad_norm=1.0,    # 使用更保守的梯度裁剪
    fp16_full_eval=True,
    seed=42,
    remove_unused_columns=False
)

# ========== 🚀 6. 动态填充的 DataCollator ==========
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8
)

# ========== 🚀 7. 初始化 Trainer ==========
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    data_collator=data_collator
)

# ========== 🚀 8. 启动训练 ==========
trainer.train()

# ========== 🚀 9. 保存适配器 ==========
model.save_pretrained("./output/GenreShift_lora/final")

# ========== 🚀 10. 推理测试 ==========
eval_model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
eval_model = PeftModel.from_pretrained(eval_model, "./output/GenreShift_lora/final")
eval_model.eval()

def generate_response(prompt):
    # 强制添加身份提示在每次推理前
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(eval_model.device)
    
    outputs = eval_model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,      # 极低温度促进确定性回答
        top_p=0.9,
        repetition_penalty=1.2,  # 增强重复惩罚
        top_k=40  # 限制词汇选择范围
    )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# 测试几个核心问题
test_prompts = [
    "请介绍一下你自己，你是什么项目？",
    "你是通义千问吗？",
    "GenreShift能处理哪些体裁的转换？",
    "你的团队成员有哪些人？"
]

for prompt in test_prompts:
    print(f"问题: {prompt}")
    response = generate_response(prompt)
    print(f"回答: {response}\n")