# coding=utf-8
# Copyright 2025 Toshihiko Aoki.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# origin: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

# tested jetson container: dustynv/vllm:r36.4.0-cu128

import json
import re

from dataclasses import dataclass
from typing import Literal
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

# for jetson comment out
# /usr/local/lib/python3.10/dist-packages/unsloth/__init__.py
# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
#    "expandable_segments:True,"\
#    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
from unsloth import FastLanguageModel, PatchFastRL


def is_bfloat16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@dataclass
class GRPOTrainingArguments:
    """Training configuration for HuggingFace models."""
    use_vllm: bool = True
    gpu_memory_utilization = 0.4
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: Literal["cosine"] = "cosine"
    optim: Literal["paged_adamw_8bit"] = "paged_adamw_8bit"
    logging_steps: int = 1
    bf16: bool = is_bfloat16_supported()
    fp16: bool = not is_bfloat16_supported()
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6
    max_prompt_length: int = 256
    max_completion_length: int = 2048
    num_train_epochs: int = 1
    save_steps: int = 10
    max_grad_norm: float = 0.1
    report_to: Literal["none"] = "none"
    output_dir: str = "outputs"
    max_seq_length = 32768
    lora_rank = 64
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

from transformers import HfArgumentParser, TrainingArguments

parser = HfArgumentParser(GRPOTrainingArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Can increase for longer reasoning traces
# Larger rank = smarter, but slower
PatchFastRL("GRPO", FastLanguageModel)

# Enable vLLM fast inference
# False for LoRA 16bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=script_args.model_name,
    max_seq_length=script_args.max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=script_args.lora_rank,
    gpu_memory_utilization=script_args.gpu_memory_utilization,
)


model = FastLanguageModel.get_peft_model(
    model,
    # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    r=script_args.lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Remove QKVO if out of memory
    lora_alpha=script_args.lora_rank,
    # Enable long context finetuning
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

SYSTEM_PROMPT = """
以下の形式で応答しなさい:
<think>
...
</think>
<answer>
...
</answer>
"""


def extract_fields(record):
    # questionの前処理
    question = record.get("question", "")

    # "Question:\n"が先頭にあれば削除
    prefix = "Question:\n"
    if question.startswith(prefix):
        question = question[len(prefix):]

    # "\nAnswer:"以降の部分を削除
    suffix_marker = "\nAnswer:"
    idx = question.find(suffix_marker)
    if idx != -1:
        question = question[:idx]

    # response, answerの取得
    response = record.get("response", "")
    answer = record.get("answer", "")

    return question, response, answer


def extract_last_number(s):
    matches = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", s)
    return float(matches[-1].replace(',', '')) if matches else None


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()



def load_mcot(json_filepath='mcot_math.json', lang='ja', jsonl=False) -> Dataset:
    import pandas as pd
    records = []
    if jsonl:
        with open(json_filepath, "r", encoding="utf-8") as f:
            # outputs/mcot_ft_mgsm_ja.json 250 length
            for i, line in enumerate(f):
                record = json.loads(line)
                question, response, answer = extract_fields(record)
                records.append(
                    {
                        'question': question,
                        'answer': answer,
                        "prompt": []
                    }
                )

    else:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if item['lang'] == lang:
                    records.append(
                        {
                            'question': item['question'],
                            'answer': extract_last_number(item['answer']),
                            "prompt": []
                        }
                    )

    df = pd.DataFrame(records)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer']
    })

    return data

train_dataset = load_mcot()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]

    print('-'*20,
          f"Question: \n{q}",
          f"\nAnswer: \n{answer[0]}",
          f"\nResponse: \n{responses[0]}",
          f"\nExtracted: \n{extracted_responses[0]}"
          f"\nExtracted last Number: \n{extract_last_number(extracted_responses[0])}"
    )
    return [2.0 if r == a else 0.0 for r, a in zip([extract_last_number(s) for s in extracted_responses], answer)]



def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


training_args = GRPOConfig(
    use_vllm=script_args.use_vllm,
    learning_rate=script_args.learning_rate,
    adam_beta1=script_args.adam_beta1,
    adam_beta2=script_args.adam_beta2,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    optim=script_args.optim,
    logging_steps=script_args.logging_steps,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    num_generations=script_args.num_generations,
    max_prompt_length=script_args.max_prompt_length,
    max_completion_length=script_args.max_completion_length,
    num_train_epochs=script_args.num_train_epochs,
    save_steps=script_args.save_steps,
    max_grad_norm=script_args.max_grad_norm,
    report_to=script_args.report_to,
    output_dir=script_args.output_dir,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train(
    # resume_from_checkpoint=True
)
model.save_lora("grpo_saved_lora")

text = tokenizer.apply_chat_template([
    {
        "role": "user",
        "content": "円周率を計算して"
    },
], tokenize=False, add_generation_prompt=True)


from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = model.fast_generate(
    [text],
    sampling_params=sampling_params,
    lora_request=None,
)[0].outputs[0].text
print(output)


text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "円周率を計算して"},
], tokenize=False, add_generation_prompt=True)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
print(output)

# Just LoRA adapters
model.save_pretrained_merged(
    "model_lora", tokenizer, save_method="lora")

# Merge to 16bit
model.save_pretrained_merged(
    "model16", tokenizer, save_method="merged_16bit")

