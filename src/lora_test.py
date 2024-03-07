import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from peft import AutoPeftModelForCausalLM
from template import templates_lookup


@dataclass
class ScriptArguments:
    max_seq_length: Optional[int] = field(default=300)
    use_nai_tokenizer: Optional[bool] = field(
        default=False,
    )
    slow_tokenizer: Optional[bool] = field(
        default=False,
    )
    load_in_4bit: Optional[bool] = field(
        default=False,
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
    )
    lora_model: str = field(
        default="lora_model/final_checkpoints",
        metadata={"help": "apply lora model directory"},
    )
    base_model: str = field(
        default=None,
    )
    add_bos_token: Optional[bool] = field(
        default=True,
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
    )
    use_sdpa: Optional[bool] = field(
        default=True,
    )
    bf16: Optional[bool] = field(
        default=True,
    )
    prompt_format: str = field(
        default="alpaca_short",
        metadata={"help": "lookup template.py"},
    )
    do_sample: Optional[bool] = field(
        default=True,
    )
    tokenizer_model: str = field(
        default=None,
        metadata={"help": "apply tokenizer model directory or path"},
    )
    use_unsloth:str = field(
        default=False
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

instruct_template = templates_lookup.get(script_args.prompt_format)
if instruct_template is None:
    raise ValueError("not found prompt_format :", script_args.prompt_format)

# require flash-attn or torch 2.1 later
attn_impl = "eager"
if script_args.use_sdpa:
    attn_impl = "sdpa"
if script_args.use_flash_attention_2:
    attn_impl = "flash_attention_2"

tokenizer = None
if script_args.base_model:
    if script_args.use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=script_args.base_model,
            dtype=None,
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        FastLanguageModel.for_inference(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model,
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit,
            device_map="auto",
            torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
            attn_implementation=attn_impl,
            trust_remote_code=True)
        if not script_args.load_in_4bit and not script_args.load_in_8bit:
            model.cuda()

    script_args.lora_model = script_args.base_model
else:
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.lora_model,
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
        is_trainable=False,
        attn_implementation=attn_impl,
        device_map="auto",
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16,
        trust_remote_code=True)
    model.cuda()

if model.config.model_type == 'gemma':
    from transformers import activations
    model.act_fn = activations.ACT2FN["gelu_pytorch_tanh"]

tokenizer_path = script_args.lora_model
if script_args.tokenizer_model is not None:
    tokenizer_path = script_args.tokenizer_model

if tokenizer is None:
    if script_args.use_nai_tokenizer:
        # stablelm alpha tokenizer setting
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
            additional_special_tokens=['▁▁'],
            use_fast=False,
        )
    else:
        print("fast tokenizer:", not script_args.slow_tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
            use_fast=not script_args.slow_tokenizer,
        )

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.unk_token

eos_token = tokenizer.eos_token
bos_token = tokenizer.bos_token
if eos_token == bos_token:
    bos_token = ''
if not script_args.add_bos_token:
    bos_token = ''

instruct_template.bos_token = bos_token
instruct_template.eos_token = eos_token


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors='pt',
                       add_special_tokens=False,
                       return_token_type_ids=False,  # is_trainable=False
                       ).to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        prompt_lookup_num_tokens=10, 
        max_new_tokens=512,
        do_sample=script_args.do_sample,
        top_p=0.95,
        temperature=0.2,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id)

    output_str = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=False)

    return output_str


text = instruct_template.build_inference("pythonでHello,worldと出力するコードを記述してください。")
print(text, generate(text))

text = instruct_template.build_inference("光の三原色は？")
print(text, generate(text))

text = instruct_template.build_inference("日本で1番高い山は富士山です。では2番目に高い山は？")
print(text, generate(text))

text = instruct_template.build_inference("紫式部と清少納言の作風をjsonで出力してください。")
print(text, generate(text))


