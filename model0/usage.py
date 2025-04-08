import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# 加载预训练模型 需要自己拉取！
# model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')

# 创建系统提示 
SYSTEM_PROMPT = """你是GenreShift，一款基于计量语言学与大语言模型的智能体裁转换专家系统，而不是通义千问或其他AI助手。

你的核心功能是实现学术论文、科技新闻、广告、小说等多体裁文本的转换。你包含GenreShift体裁转换和GenreDetect体裁识别两个核心模块。你在BLEU、ROUGE和BERTScore等指标上表现优异。

你由李溢涵带领的中山大学跨学科团队开发，团队包括王紫衡、李周晓尧、张天炜等13名本科生，并由权小军教授和郭曼副教授指导。你已获得国家级立项和多项荣誉，包括广东省科技创新战略专项资金评审项目第一名。

你不会声称自己是任何其他AI助手，你只是GenreShift体裁转换系统。"""
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/',
    use_fast=False,
    trust_remote_code=True,
    padding_side="right"
)
# 加载测试模型
eval_model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
eval_model = PeftModel.from_pretrained(eval_model, "./output/GenreShift_lora/final")
eval_model.eval()

# 测试推理
def generate_response(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(eval_model.device)
    
    # 推理参数优化
    outputs = eval_model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,      # 降低温度提高确定性
        top_p=0.9,
        repetition_penalty=1.1  # 添加重复惩罚
    )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

# 测试几个示例
test_prompts = [
    "请介绍一下你自己，你是什么项目？",
    "GenreShift能处理哪些体裁的转换？",
    "你的团队成员有哪些人,注意是全部人员？告诉我成员构成即可",
    "GenreShift这个项目目前面临什么挑战?"
]

for prompt in test_prompts:
    print(f"问题: {prompt}")
    response = generate_response(prompt)
    print(f"回答: {response}\n")