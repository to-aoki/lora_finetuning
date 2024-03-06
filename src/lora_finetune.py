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
# written: Toshihiko Aoki

import os
import re
import math
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    PreTrainedModel
)
from accelerate import Accelerator
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import bitsandbytes as bnb
from peft import LoraConfig, LoftQConfig
from trl import SFTTrainer
from template import templates_lookup

# This example lora fine-tunes Llama v2 model on Jetson AGX Orin
# Versions used:
# accelerate == 0.25.0
# peft == 0.8.2
# bitsandbytes == 0.41.2
# transformers == 4.37.2
# trl == 0.7.10
# xformers == 0.0.25+d93b3fd.d20240211


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
    base_model: Optional[str] = field(
        default="elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[List[str]] = field(
        default_factory=lambda: ['sakusakumura/databricks-dolly-15k-ja-scored'],
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 8bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
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
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_8bit",
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
    neftune_noise_alpha: Optional[float] = field(default=5.)
    add_bos_token: Optional[bool] = field(
        default=True,
    )
    prompt_format: str = field(
        default="alpaca_short",
        metadata={"help": "lookup template.py"},
    )
    use_flash_attention_2: bool = field(
        default=False,
    )
    use_sdpa: bool = field(
        default=True,
    )
    target_all_layer: bool = field(
        default=False,
    )
    only_instruct: bool = field(
        default=True,
    )
    report_to: str = field(
        default="none",
    )
    long_lora: bool = field(
        default=False,
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    dolly_ja_score: float = field(
        default=0.9,
    )
    with_unsloth: bool = field(
        default=False,
    )
    with_loftq: bool = field(
        default=False,  # initialize slow
    )
    origin_tokenizer: bool = field(
        default=True,
    )
    use_hf_auth_token: bool = field(
        default=False,
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

instruct_template = templates_lookup.get(script_args.prompt_format)


accelerator = Accelerator()
def find_all_linear_names(target_model):
    # https://note.com/npaka/n/na506c63b8cc9#260d93c9-2984-4a34-8118-8f68d64655b6
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

    loftq_config = {}
    init_lora_weights = True
    if args.with_loftq:
        init_lora_weights = 'loftq'
        loftq_config = LoftQConfig(loftq_bits=4 if args.use_4bit else 8, loftq_iter=1)
    elif not args.with_unsloth:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = "auto"

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if config.model_type == 'gemma':
        # https://github.com/huggingface/transformers/pull/29402
        config.hidden_act = 'gelu_pytorch_tanh'

    if args.with_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=False if args.with_loftq else args.use_4bit,
            load_in_8bit=False if args.with_loftq or args.use_4bit else True,
        )

    else:
        # require flash-attn or torch 2.1 later
        attn_impl = None
        if args.use_sdpa:
            attn_impl = "sdpa"
        if args.use_flash_attention_2:
            # RuntimeError: FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800
            # FlashAttention v2.5.5 support
            attn_impl = "flash_attention_2"

        if config.model_type == 'gpt2':
            attn_impl = "eager"

        if args.long_lora and (config.model_type == "gpt-neox" or config.model_type == "llama"):
            print('with long_lora')
            orig_rope_scaling = getattr(config, "rope_scaling", None)
            if orig_rope_scaling is None:
                orig_rope_scaling = {"factor": 1}
            orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len is not None:
                print('max_position_embeddings:', orig_ctx_len)
                orig_ctx_len *= orig_rope_scaling_factor
                if args.max_seq_length > orig_ctx_len:
                    scaling_factor = float(math.ceil(args.max_seq_length / orig_ctx_len))
                    print('scaling_factor:', scaling_factor)
                    config.rope_scaling = {"type": "linear", "factor": scaling_factor}
                    if config.model_type == "gpt-neox":
                        print("modify long tokens: gpt-neox")
                        # https://github.com/dvlab-research/LongLoRA/blob/main/gptneox_attn_replace.py
                        from gptneox_attn_replace import replace_gpt_neox_attn
                        replace_gpt_neox_attn(args.use_flash_attention_2, args.use_full_attn)
                    elif config.model_type == "llama":
                        print("modify long tokens: llama")
                        # https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace_sft.py
                        from llama_attn_replace_sft import replace_llama_attn
                        replace_llama_attn(args.use_flash_attention_2, args.use_full_attn)
                    config.max_position_embeddings = args.max_seq_length
                    config.save_pretrained(args.output_dir)

        if bool(re.match(r'.*japanese-stablelm.*alpha.*', script_args.base_model)):
            print("ja-stablelm-alpha")
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                "novelai/nerdstash-tokenizer-v1",
                use_fast=False,  # lm-evaluate scripts use_fast=False
                trust_remote_code=True,
                additional_special_tokens=['▁▁']
            )
            print('nai tokenizer loaded')
        elif script_args.base_model.startswith(("rinna/", "line-corporation/japanese-large")):
            tokenizer = AutoTokenizer.from_pretrained(
                script_args.base_model,
                use_fast=False,
            )
            print('use_fast=False tokenizer loaded')
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                script_args.base_model,
                use_fast=True,
                trust_remote_code=True,
            )
        if args.with_loftq:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                config=config,
                device_map=device_map,
                use_auth_token=args.use_hf_auth_token,
                torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=bnb_config,
                config=config,
                device_map=device_map,
                use_auth_token=args.use_hf_auth_token,
                torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        print(model)

    if model.config.model_type == 'llama':
        # https://www.docswell.com/s/KanHatakeyama/ZYW6ME-2023-12-09-121017#p31
        target_modules = [
            "lm_head",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
        ]
    elif model.config.model_type == 'phi-msft':
        target_modules = ['lm_head.linear', 'transformer.embd.wte']
    elif model.config.model_type == 'gpt2':
        # llm-jp
        target_modules = [
            "c_attn",
            "c_proj",
            "c_fc",
        ]
    elif model.config.model_type == 'gemma':
        target_modules = [
            "k_proj",
            "up_proj",
            "v_proj",
            "gate_proj",
            "o_proj",
            "q_proj",
            "down_proj"
        ]
    else:
        target_modules = find_all_linear_names(model)

    fan_in_fan_out = False
    if model.config.model_type == 'gpt2':
        fan_in_fan_out = True

    if args.with_unsloth:
        # accepted_modules = frozenset
        if model.config.model_type == 'llama':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing=True,
            random_state=3407,
            max_seq_length=args.max_seq_length,
        )
        lora_config = None

    else:
        print(target_modules)

        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            fan_in_fan_out=fan_in_fan_out
        )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.unk_token

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.unk_token

    tokenizer.padding_side = "right"

    print('target_modules:', target_modules)
    print("=" * 80)
    print(tokenizer.eos_token_id, tokenizer.eos_token)
    print(tokenizer.bos_token_id, tokenizer.bos_token)
    print(tokenizer.pad_token_id, tokenizer.pad_token)
    print(tokenizer.unk_token_id, tokenizer.unk_token)
    print(tokenizer.padding_side)
    print("=" * 80)


    if script_args.base_model.startswith("meta-llama/Llama-2"):
        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    return model, lora_config, tokenizer


model, lora_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

gradient_checkpointing = script_args.gradient_checkpointing
if model.config.model_type == 'phi-msft':
    # too large (2024-01-07)
    gradient_checkpointing = False

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
    gradient_checkpointing=gradient_checkpointing,
    report_to=script_args.report_to
)


class DataCollatorForCompletion:

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id)
        )


class OnlyInstructSFTTrainer(SFTTrainer):
    r"""
       trl DataCollatorForCompletionOnlyLM is not working well.
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
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha=5,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        only_instruction: bool = False
    ):
        self.only_instruction = only_instruction
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
        )

    def _prepare_non_packed_dataloader(
        self, tokenizer, dataset, dataset_text_field, max_seq_length, formatting_func=None, add_special_tokens=False,
            remove_unused_columns=True,
    ):

        def tokenize(element):
            formatted = formatting_func(element)
            outputs = tokenizer(
                formatted[0],
                add_special_tokens=False,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
            )
            if self.only_instruction:
                instruct = tokenizer(
                    formatted[1],
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                )
                input_batch = []
                sources_tokenized_lens = []
                input_ids_lens = [len(input_id) for input_id in instruct["input_ids"]]
                for input_length, input_ids in zip(input_ids_lens, outputs["input_ids"]):
                    if input_length > max_seq_length:
                        continue
                    input_batch.append(input_ids)
                    sources_tokenized_lens.append(input_length)
                labels = copy.deepcopy(input_batch)
                for label, source_len in zip(labels, sources_tokenized_lens):
                    label[:source_len] = [-100] * source_len
                return {"input_ids": input_batch, "labels": labels}
            else:
                labels = copy.deepcopy(outputs["input_ids"])
                return {"input_ids": outputs["input_ids"], "labels": labels}

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset


eos_token = tokenizer.eos_token
bos_token = tokenizer.bos_token
if eos_token == bos_token:
    bos_token = ''
if not script_args.add_bos_token:
    bos_token = ''
instruct_template.bos_token = bos_token
instruct_template.eos_token = eos_token

instruct_columns = ["instruction", "input", "output"]

dataset = None
for dataset_name in script_args.dataset_name:
    if dataset_name.endswith('.json'):
        dataset_one = load_dataset(
            'json',
            data_files=script_args.dataset_name,
            split="train",
        )
    else:
        if os.path.exists(dataset_name):
            dataset_one = load_from_disk(dataset_name)
        else:
            dataset_one = load_dataset(dataset_name, split="train")
    if dataset_name == 'sakusakumura/databricks-dolly-15k-ja-scored':
        dataset_one = dataset_one.filter(lambda example: example["bertscore"]["f1"] > script_args.dolly_ja_score)

    dataset_one = dataset_one.shuffle(seed=42)
    if "input" not in dataset_one.column_names:
        dataset_one = dataset_one.map(lambda example: {"input": "", **example})

    dataset_one = dataset_one.map(lambda example: {col: example[col] for col in instruct_columns})

    if dataset is None:
        dataset = dataset_one
    else:
        common_columns = set(dataset.column_names) & set(dataset_one.column_names)
        dataset = dataset.map(lambda example: {col: example[col] for col in common_columns})
        dataset_one = dataset_one.map(lambda example: {col: example[col] for col in common_columns})
        dataset = concatenate_datasets([dataset, dataset_one])


neftune_noise_alpha = script_args.neftune_noise_alpha
if neftune_noise_alpha <= 0:
    neftune_noise_alpha = None


callbacks = []

if script_args.long_lora:
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            modules_to_save = ["embed", "norm"]
            state_dict = kwargs["model"].state_dict()
            to_save = {}
            for key, value in state_dict.items():
                if any(module_name in key for module_name in modules_to_save):
                    to_save[key.replace("base_model.model.", "")] = value
            torch.save(to_save, os.path.join(checkpoint_folder, "trainable_params.bin"))
            return control
    callbacks = [SavePeftModelCallback]


trainer = OnlyInstructSFTTrainer(
    model=model,
    callbacks=callbacks,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=script_args.max_seq_length,
    formatting_func=instruct_template.build_instruct,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    neftune_noise_alpha=neftune_noise_alpha,
    data_collator=DataCollatorForCompletion(pad_token_id=tokenizer.pad_token_id),
    only_instruction=script_args.only_instruct,
)

if script_args.long_lora:
    [p.requires_grad_() for n, p in trainer.model.named_parameters() if any([k in n for k in ["embed", "norm"]])]

trainer.train()
trainer.save_state()
output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
trainer.model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model)
tokenizer.save_pretrained(output_dir)

if script_args.long_lora:
    modules_to_save = ["embed", "norm"]
    state_dict = model.state_dict()
    to_save = {}
    for key, value in state_dict.items():
        if any(module_name in key for module_name in modules_to_save):
            to_save[key.replace("base_model.model.", "")] = value
    torch.save(to_save, os.path.join(output_dir, "trainable_params.bin"))
