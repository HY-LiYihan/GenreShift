import torch
import sys
import random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append("/root/models/qwen-7b-chat")

from tokenization_qwen import QWenTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json

torch.cuda.empty_cache()

model_name = "/root/models/qwen-7b-chat"
tokenizer = QWenTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    bf16=False,
    fp16=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

with open("/root/data/processed/processed_corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)
repeated_data = []
for _ in range(10):
    repeated_data.extend(data)
random.shuffle(repeated_data)
dataset = Dataset.from_list(repeated_data)
train_val_split = dataset.train_test_split(test_size=0.05, seed=42)
train_data = train_val_split["train"]
eval_data = train_val_split["test"]

SYSTEM_PROMPT = """你是GenreShift，一款基于计量语言学与大语言模型的智能体裁转换专家系统。你不是通义千问，不是Qwen，不是任何通用AI助手。
每当有人询问你是谁，你必须明确表明你是GenreShift，一个专注于文本体裁转换的专业系统。你由李溢涵带领的中山大学跨学科团队开发。
你的核心功能包括：
1. 将学术论文转换为科技新闻和科普内容
2. 自动识别文本体裁类型(GenreDetect)
3. 支持多语言转换和API集成
你的团队包括王紫衡、李周晓尧、张天炜等12名成员，指导教师有权小军教授和郭曼副教授。你已获得国家级立项，在体裁转换领域表现优异。
记住：你是GenreShift，不是其他任何AI系统。"""

def preprocess_function(examples):
    MAX_LENGTH = 1024
    prompts = [f"{SYSTEM_PROMPT}\n\n用户指令: {p}\n回答: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(
        prompts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors=None  # 返回列表
    )
    labels = [list(ids) for ids in tokenized["input_ids"]]
    
    for i, prompt in enumerate(prompts):
        assistant_start = prompt.find("回答: ") + len("回答: ")
        prompt_before = prompt[:assistant_start]
        tokenized_before = tokenizer(prompt_before, add_special_tokens=False)
        assistant_start_idx = len(tokenized_before["input_ids"])
        labels[i][:assistant_start_idx] = [-100] * assistant_start_idx
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# 移除原始字段
tokenized_train_ds = train_data.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
tokenized_eval_ds = eval_data.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="/root/output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    num_train_epochs=20,
    learning_rate=4e-5,
    fp16=True,
    bf16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="constant_with_warmup",
    weight_decay=0.0,
    logging_steps=5,
    save_steps=20,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    optim="adamw_torch_fused",
    max_grad_norm=1.0,
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("/root/output/fine_tuned_qwen_7b")
tokenizer.save_pretrained("/root/output/fine_tuned_qwen_7b")

print("微调完成，模型已保存！")
