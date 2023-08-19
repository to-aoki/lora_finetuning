from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
)
from peft import AutoPeftModelForCausalLM


@dataclass
class ScriptArguments:
    max_seq_length: Optional[int] = field(default=1024)
    use_nai_tokenizer: Optional[bool] = field(
        default=False,
    )
    slow_tokenizer: Optional[bool] = field(
        default=False,
    )
    load_in_4bit: Optional[bool] = field(
        default=True,
    )
    lora_model: str = field(
        default="lora_model/final_checkpoints",
        metadata={"help": "apply lora model directory"},
    )
    replace_line_sep: str = field(
        default=None,
        metadata={"help": "The line seperator. for rinna <NL>."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model = AutoPeftModelForCausalLM.from_pretrained(
    script_args.lora_model,
    load_in_4bit=script_args.load_in_4bit,
    is_trainable=False,
    device_map="auto", torch_dtype=torch.bfloat16,
    trust_remote_code=True)

add_special_tokens = True

if script_args.use_nai_tokenizer:
    # stable-lm tokenizer setting
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        script_args.lora_model, trust_remote_code=True,
        additional_special_tokens=['▁▁'],
        use_fast=False,
    )
    add_special_tokens = False
else:
    print("fast tokenizer:", not script_args.slow_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.lora_model, trust_remote_code=True,
        use_fast=not script_args.slow_tokenizer,
    )
    if script_args.slow_tokenizer:
        # T5Tokenizer（rinna/line）は末尾に</s>を付与してしまうのでFalseに
        add_special_tokens = False


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors='pt',
                       add_special_tokens=add_special_tokens,
                       return_token_type_ids=False,  # is_trainable=False
                       ).to(model.device)
    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=script_args.max_seq_length,
        temperature=0.7,
        top_p=0.7,
        top_k=40,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )

    token = outputs.sequences[0, input_length:]
    if tokenizer.eos_token_id in token:
        eos_index = (token == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        output_str = tokenizer.decode(token[:eos_index], skip_special_tokens=False)
    else:
        output_str = tokenizer.decode(token, skip_special_tokens=False)
    if script_args.replace_line_sep is not None:
        output_str = output_str.replace(script_args.replace_line_sep, "\n")

    return output_str


text = f"以下は、ある作業を記述した指示です。要求を適切に満たすような応答を書きなさい。\n\n### 指示:\n光の三原色は？\n\n### 応答:\n"
print(generate(text))
text = f"以下は、ある作業を記述した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n日本で1番高い山は富士山です。では2番目に高い山は？\n\n### 応答:\n"
print(generate(text))
text = f"以下は、ある作業を記述した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示:\n紫式部と清少納言の作風を表で比較してください。\n\n### 応答:\n"
print(generate(text))
