# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# origin: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
import bitsandbytes as bnb
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer


# This example lora fine-tunes Llama v2 model on Jetson AGX Orin
#
# Versions used:
# accelerate == 0.21.0
# peft == 0.4.0
# bitsandbytes == 0.39.0
# transformers == 4.31.0
# trl == 0.5.0

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=3e-5)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_r: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default="stabilityai/japanese-stablelm-base-alpha-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="kunishou/databricks-dolly-15k-ja",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_linear_target: Optional[bool] = field(
        default=False,
        metadata={"help": "Training linear targets."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",  # low memory for paged_adamw_8bit
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=-1, metadata={"help": "How many optimizer update steps to take"})

    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    weight_decay: float = field(default=0.001)
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./lora_model",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    replace_line_sep: str = field(
        default=None,
        metadata={"help": "The line seperator. for rinna-3.6b specific <NL>."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def find_all_linear_names(target_model):
    # https://note.com/npaka/n/na506c63b8cc9#260d93c9-2984-4a34-8118-8f68d64655b6
    #
    cls = bnb.nn.Linear4bit  # (default:torch.nn.Linear,4bit:bnb.nn.Linear4bit,8bit:bnb.nn.Linear8bitLt)
    lora_module_names = set()
    for name, module in target_model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8 and not script_args.bf16:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=True,
        trust_remote_code=True,
    )

    if script_args.model_name.startswith(("meta-llama/Llama-2", "elyza/ELYZA-japanese-Llama-2")):
        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens",
                      "lm_head"]
    if script_args.use_linear_target:
        target_modules = find_all_linear_names(model)
    print('target_modules:', target_modules)

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    if script_args.model_name.startswith("stabilityai/japanese-stablelm-base"):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            use_fast=False,  # lm-evaluate scripts use_fast=False
            trust_remote_code=True,
            additional_special_tokens=['▁▁']
        )
        print('nai tokenizer loaded')
    elif script_args.model_name.startswith(("rinna/","line-corporation/japanese-large")):
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name,
            use_fast=False,
        )
        print('use_fast=False tokenizer loaded')
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name, trust_remote_code=True,
        )

    if script_args.model_name.startswith("matsuo-lab/"):
        tokenizer.eos_token_id = 0
        tokenizer.pad_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)

    if script_args.model_name.startswith("mistralai/Mistral-7B"):
        # normalized Trueになってるのでトークナイザで強制する。multi-turn都合などで文字列concatする際はtokenizer_config.jsonいじる
        # https://github.com/huggingface/transformers/issues/23818
        tokenizer.add_eos_token = True

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 80)
    print(tokenizer.eos_token_id, tokenizer.eos_token)
    print(tokenizer.bos_token_id, tokenizer.bos_token)
    print(tokenizer.pad_token_id, tokenizer.pad_token)
    print(tokenizer.unk_token_id, tokenizer.unk_token)
    print("=" * 80)

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    num_train_epochs=script_args.num_train_epochs,
    warmup_ratio=script_args.warmup_ratio,
    weight_decay=script_args.weight_decay,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
dataset = load_dataset(script_args.dataset_name, split="train")
eos_token = tokenizer.eos_token
if script_args.model_name.startswith("mistralai/Mistral-7B"):
    eos_token = ''

def formatting_prompts_func_alpaca_short(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        if example['input'][i]:
            text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}" + eos_token
        else:
            text = f"### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_rinna_ja(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        if example['input'][i]:
            text = f"ユーザー: {example['instruction'][i]}\nシステム: 分かりました。\nユーザー: {example['input'][i]}\nシステム: {example['output'][i]}" + eos_token
        else:
            text = f"ユーザー: {example['instruction'][i]}\nシステム: {example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_alpaca_ja(example):

    output_texts = []
    for i in range(len(example['instruction'])):
        if example['input'][i]:
            text = f"以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 入力:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token
        else:
            text = f"以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_dolly_categories(example):

    output_texts = []
    for i in range(len(example['instruction'])):
        category = example['category']
        if category == "open_qa":
            text = f"以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "closed_qa":
            text = f"以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を入力から抜き出して書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 入力:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "general_qa":
            text = f"下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "classification":
            text = f"以下は、ある作業を説明した指示です。指示に与えられる選択肢から応答を書きなさい。\n\n### 指示\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "information_extraction":
            text = f"以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を入力より抽出して書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 入力:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "brainstorming":
            text = f"以下は、ある議題に関する意見の収集です。指示に従い応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "summarization":
            text = f"以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を入力を要約して書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 入力:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        elif category == "creative_writing":
            text = f"以下は、あるストーリーの方針を説明した指示です。指示を満たすストーリーを応答に書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token

        else:
            if example['input'][i]:
                text = f"以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 入力:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token
            else:
                text = f"以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n{example['instruction'][i]}\n\n### 応答:\n{example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_llama2_chat(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        if example['input'][i]:
            text = f"[INST] <<SYS>>\nあなたは親切で、礼儀正しく、誠実なアシスタントです。 常に安全を保ちながら、できるだけ役立つように答えてください。 " \
                f"回答には、有害、非倫理的、人種差別的、性差別的、有毒、危険、または違法なコンテンツを含めてはいけません。 回答は社会的に偏見がなく、本質的に前向きなものであることを確認してください。 " \
                f"質問には<<input>>を参照して答えてください。" \
                f"質問が意味をなさない場合、または入力に一貫性がない場合は、正しくないことに答えるのではなく、その理由を説明してください。 質問の答えや<<input>>がわからない場合は、誤った情報を共有しないでください。" \
                f"\n<</SYS>>\n\n<<input>>\n{example['input'][i]}\n<</input>>\n\n{example['instruction'][i]} [/INST] {example['output'][i]}" + eos_token
        else:
            text = f"[INST] <<SYS>>\nあなたは親切で、礼儀正しく、誠実なアシスタントです。 常に安全を保ちながら、できるだけ役立つように答えてください。 " \
                   f"回答には、有害、非倫理的、人種差別的、性差別的、有毒、危険、または違法なコンテンツを含めてはいけません。 回答は社会的に偏見がなく、本質的に前向きなものであることを確認してください。 " \
                   f"質問が意味をなさない場合、または事実に一貫性がない場合は、正しくないことに答えるのではなく、その理由を説明してください。 質問の答えがわからない場合は、誤った情報を共有しないでください。" \
                   f"\n<</SYS>>\n\n{example['instruction'][i]} [/INST]{example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_llama2_chat_wo_role(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        if example['input'][i]:
            text = f"[INST] <<SYS>>\n以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。<<input>>を適切に満たすような応答を書きなさい。" \
                f"\n<</SYS>>\n\n<<input>>\n{example['input'][i]}\n<</input>>\n\n{example['instruction'][i]} [/INST] {example['output'][i]}" + eos_token
        else:
            text = f"[INST] <<SYS>>\n以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。" \
                   f"\n<</SYS>>\n\n{example['instruction'][i]} [/INST]{example['output'][i]}" + eos_token
        if script_args.replace_line_sep is not None:
            text = text.replace('\n', script_args.replace_line_sep)
        output_texts.append(text)
    return output_texts


import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union
from datasets import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback


class WithoutSpecialTokensSFTTrainer(SFTTrainer):
    r"""
       Class definition of the Supervised Finetuning Trainer (SFT Trainer) for japaneese stablelm.
       Only changed tokenizer handling of special tokens from the original.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = False,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            peft_config,
            dataset_text_field,
            packing,
            formatting_func,
            max_seq_length,
            infinite,
            num_of_sequences,
            chars_per_token,
            # v 0.5.1
            # dataset_num_proc,
            # dataset_batch_size,
        )

    def _prepare_non_packed_dataloader(
        self, tokenizer, dataset, dataset_text_field, max_seq_len, formatting_func=None
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=False,  # only FIX
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            # num_proc=self.dataset_num_proc,
            # batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset


trainer_class = SFTTrainer
if script_args.model_name.startswith(
        ("rinna/", "stabilityai/japanese-stablelm-base", "line-corporation/japanese-large")):
    print('without special tokens')
    trainer_class = WithoutSpecialTokensSFTTrainer


"""
rinna, line: T5Tokenizer(use_fast=False), eos(</s>) token append default.
stablelm: LlamaTokenizer, bos token(<|startoftext|>) append default. no used pre-training?
llama2: LlamaTokenizer, used bos and eos.
opencalm: no special tokens append.
matsuo-lab: no special tokens append.
"""

trainer = trainer_class(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    formatting_func=formatting_prompts_func_llama2_chat_wo_role,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()
output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
