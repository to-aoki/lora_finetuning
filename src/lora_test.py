import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    LogitsProcessor,
    AutoModelForCausalLM
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
        default=True,
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
    add_special_tokens: Optional[bool] = field(
        default=True,
    )
    add_bos_token: Optional[bool] = field(
        default=True,
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
    )
    bf16: Optional[bool] = field(
        default=True,
    )
    prompt_format: str = field(
        default="deepseek_coder",
        metadata={"help": "lookup template.py"},
    )
    do_sample: Optional[bool] = field(
        default=True,
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

instruct_template = templates_lookup.get(script_args.prompt_format)

attn_impl = "flash_attention_2" if script_args.use_flash_attention_2 else 'sdpa'

if script_args.base_model:
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

add_special_tokens = script_args.add_special_tokens
if script_args.use_nai_tokenizer:
    # stablelm alpha tokenizer setting
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

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

eos_token = tokenizer.eos_token
bos_token = tokenizer.bos_token
add_special_tokens = script_args.add_special_tokens
test_message = '今日もいい天気ですね'
test_ids = tokenizer(test_message, add_special_tokens=add_special_tokens)

if add_special_tokens:
    if test_ids['input_ids'][-1] == tokenizer.eos_token_id:
        print('without special tokens')
        add_special_tokens = False
    if test_ids['input_ids'][0] == tokenizer.bos_token_id:
        bos_token = ''

if not script_args.add_bos_token:
    bos_token = ''

print("=" * 80)
print(test_message)
print(test_ids)
print('bos_token:', bos_token)
print('eos_token:', eos_token)
print('add_special_tokens:', add_special_tokens)
print(bos_token + test_message + eos_token,
      tokenizer(bos_token + test_message + eos_token, add_special_tokens=add_special_tokens))
print("=" * 80)

instruct_template.bos_token = bos_token
instruct_template.eos_token = eos_token

class EosTokenRewardLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id: int, max_length: int):
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        # start to increese the reward of the  eos_tokekn from 80% max length  progressively on length
        for cur_len in (max(0, int(self.max_length * 0.8)), self.max_length):
            ratio = cur_len / self.max_length
            num_tokens = scores.shape[1]  # size of vocab
            scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = \
                scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] * ratio * 10 * torch.exp(
                    -torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
            scores[:, self.eos_token_id] = -100
        return scores


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors='pt',
                       add_special_tokens=add_special_tokens,
                       return_token_type_ids=False,  # is_trainable=False
                       ).to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=script_args.do_sample,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id)

    output_str = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True)

    return output_str

text = instruct_template.build_inference("pythonでHello,worldと出力するコードを記述してください。")
print(text, generate(text))

text = instruct_template.build_inference("光の三原色は？")
print(text, generate(text))

text = instruct_template.build_inference("日本で1番高い山は富士山です。では2番目に高い山は？")
print(text, generate(text))

text = instruct_template.build_inference("紫式部と清少納言の作風をjsonで出力してください。")
print(text, generate(text))

