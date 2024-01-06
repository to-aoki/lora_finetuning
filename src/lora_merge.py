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
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, AutoConfig


@dataclass
class ScriptArguments:
    merge_target_path: Optional[str] = field(
        default="lora_model/final_checkpoints",
    )
    output_path: Optional[str] = field(
        default="final_merged_checkpoint",
    )
    safe_serialization: Optional[bool] = field(
        default=False,
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    base_model: Optional[str] = field(
        default=None,
    )
    base_config_path: Optional[str] = field(
        default=None,
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.base_model is not None and script_args.base_config_path is not None:
    config = AutoConfig.from_pretrained(script_args.base_config_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        config=config,
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
        device_map="auto",
    )
    trainable_params = os.path.join(script_args.merge_target_path, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        base_model.load_state_dict(torch.load(trainable_params, map_location=base_model.device), strict=False)
    model = PeftModel.from_pretrained(
        base_model,
        script_args.merge_target_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
    )
else:
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.merge_target_path, device_map="auto",
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16
    )

model = model.merge_and_unload()

model.save_pretrained(script_args.output_path, safe_serialization=script_args.safe_serialization)
tokenizer = AutoTokenizer.from_pretrained(
    script_args.merge_target_path,
)
tokenizer.save_pretrained(script_args.output_path)
