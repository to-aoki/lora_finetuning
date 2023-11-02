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
    lora_model: str = field(
        default="lora_model/final_checkpoints",
        metadata={"help": "apply lora model directory"},
    )
    base_model: str = field(
        default=None,
    )
    replace_line_sep: str = field(
        default=None,
        metadata={"help": "The line seperator. for rinna <NL>."},
    )
    add_special_tokens: Optional[bool] = field(
        default=True,
    )
    add_bos_token: Optional[bool] = field(
        default=True,
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


if script_args.base_model:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        load_in_4bit=script_args.load_in_4bit,
        device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    script_args.lora_model = script_args.base_model
else:
    model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.lora_model,
        load_in_4bit=script_args.load_in_4bit,
        is_trainable=False,
        device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True)

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

test_message = '今日もいい天気ですね'
test_ids = tokenizer(test_message, add_special_tokens=add_special_tokens)
print()
bos_token = tokenizer.bos_token
if add_special_tokens:
    if test_ids['input_ids'][-1] == tokenizer.eos_token_id:
        add_special_tokens = False
    elif test_ids['input_ids'][0] == tokenizer.bos_token_id:
        bos_token = ''
if not script_args.add_bos_token:
    bos_token = ''



print("=" * 80)
print("add_bos_token:", script_args.add_bos_token)
print("add_special_tokens:", add_special_tokens)

test_ids = tokenizer(bos_token + test_message, add_special_tokens=add_special_tokens)
print(bos_token + test_message, test_ids)

print(tokenizer.eos_token_id, tokenizer.eos_token)
print(tokenizer.bos_token_id, tokenizer.bos_token)
print(tokenizer.pad_token_id, tokenizer.pad_token)
print(tokenizer.unk_token_id, tokenizer.unk_token)
print("=" * 80)


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
        do_sample=True,
        max_new_tokens=script_args.max_seq_length,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
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


def alpaca_instruct(input_str, bos_token=bos_token):
    return bos_token + f"以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n### 指示: \n{input_str}\n\n### 応答: \n"


def llama2_instruct(input_str, bos_token=bos_token):
    return bos_token + f"[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。" \
           f"\n<</SYS>>\n\n{input_str} [/INST]"


def llm_jp_instruct(input_str, bos_token=bos_token):
    return bos_token + f"### 指示：以下の質問に答えなさい。 ### 質問：{input_str} ### 回答："


def calm2_instruct(input_str, bos_token=bos_token):
    return bos_token + f"USER: {input_str}\nASSISTANT: "


text = alpaca_instruct("光の三原色は？", bos_token)
print(generate(text))

text = alpaca_instruct("日本で1番高い山は富士山です。では2番目に高い山は？", bos_token)
print(generate(text))

text = alpaca_instruct("紫式部と清少納言の作風を表で比較してください。", bos_token)
print(generate(text))

