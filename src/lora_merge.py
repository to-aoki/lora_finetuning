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
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class ScriptArguments:
    merge_target_path: Optional[str] = field(
        default="lora_model/final_checkpoints",
    )
    output_path: Optional[str] = field(
        default="final_merged_checkpoint",
    )
    without_safe_serialization: Optional[bool] = field(
        default=False,
    )
    base_model: Optional[str] = field(
        default=None,
    )
    base_config_path: Optional[str] = field(
        default=None,
    )
    use_unsloth: Optional[bool] = field(
        default=False,
    )
    unsloth_max_seq_length: Optional[int] = field(
        default=8192  # RoPE max Scaling length
    )
    load_in_4bit: Optional[bool] = field(
        default=False,
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
    )
    unsloth_save_method: Optional[str] = field(
        default="merged_16bit",  # or merged_4bit
    )
    use_unsloth_4bit: Optional[bool] = field(
        default=False,
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    device_map: Optional[str] = field(
        default="auto",
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
tokenizer = None
if script_args.use_unsloth:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=script_args.merge_target_path,
        max_seq_length=script_args.unsloth_max_seq_length,
        dtype=None,
        load_in_4bit=script_args.use_unsloth_4bit,
    )
else:
    bnb_config = None
    if script_args.load_in_4bit or script_args.load_in_4bit:
        compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.use_nested_quant,
        )
    base_model_name_or_path = script_args.base_model
    if base_model_name_or_path is None:
        base_model_name_or_path = PeftConfig.from_pretrained(
            script_args.merge_target_path).base_model_name_or_path

    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            device_map=script_args.device_map,
            torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map=script_args.device_map,
            torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
            trust_remote_code=True,
        )

    trainable_params = os.path.join(script_args.merge_target_path, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
    model = PeftModel.from_pretrained(
        model,
        script_args.merge_target_path,
        is_trainable=False,
        device_map=script_args.device_map,
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
        trust_remote_code=True
    )


if script_args.use_unsloth:
    save_method = script_args.unsloth_save_method
    if script_args.use_unsloth_4bit and save_method != 'merged_4bit_forced':
        save_method = 'merged_4bit'
    model.save_pretrained_merged(
        script_args.output_path, tokenizer, save_method=save_method)

else:
    model = model.merge_and_unload()
    if script_args.load_in_4bit:
        # TODO 8bit
        import bitsandbytes as bnb
        from peft.utils import _get_submodules
        from bitsandbytes.functional import dequantize_4bit
        import copy
        cls = bnb.nn.Linear4bit
        dtype = torch.bfloat16 if script_args.bf16 else torch.float16
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    print(f"Dequantizing `{name}`...")
                    quant_state = copy.deepcopy(module.weight.quant_state)
                    #print(quant_state)
                    # quant_state[2] = dtype
                    weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                    new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                    new_module.weight = torch.nn.Parameter(weights)
                    new_module.to(device=script_args.device_map, dtype=dtype)

                    parent, target, target_name = _get_submodules(model, name)
                    setattr(parent, target_name, new_module)

            model.is_loaded_in_4bit = False
    model.save_pretrained(script_args.output_path,
                          safe_serialization=not script_args.without_safe_serialization)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path
        )
    tokenizer.save_pretrained(script_args.output_path)
