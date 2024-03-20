
import copy
from transformers import AutoTokenizer

DEFALUT_BOS_TOKEN = "<s>"
DEFALUT_EOS_TOKEN = "</s>"
DEFALUT_RESPONSE_PREFIX = "### Response:\n"
DEFAULT_INPUT_TEMPLATE = "### Instruction:\n{}\n### Input:\n{}\n"
DEFAULT_NO_INPUT_TEMPLATE = "### Instruction:\n{}\n"
DEFAULT_CONVERSATION_SYS = DEFAULT_NO_INPUT_TEMPLATE
DEFAULT_CONVERSATION_TEMPLATE = DEFAULT_NO_INPUT_TEMPLATE
DEFAULT_RESPONSE_SUFFIX = ""
DEFAULT_DATA_INSTRUCTION_ATTR = "instruction"
DEFAULT_DATA_OUTPUT_ATTR = "output"
DEFAULT_DATA_INPUT_ATTR = "input"


def count_placeholders(format_string):
    placeholders = 0
    in_brace = False
    for char in format_string:
        if char == "{":
            in_brace = True
        elif char == "}" and in_brace:
            placeholders += 1
            in_brace = False
    return placeholders


class InputTemplate:
    def __init__(
        self,
        bos_token=DEFALUT_BOS_TOKEN,
        eos_token=DEFALUT_EOS_TOKEN,
        input_template=DEFAULT_INPUT_TEMPLATE,
        no_input_template=DEFAULT_NO_INPUT_TEMPLATE,
        conversation_sys=DEFAULT_CONVERSATION_SYS,
        conversation_template=DEFAULT_CONVERSATION_TEMPLATE,
        response_prefix=DEFALUT_RESPONSE_PREFIX,
        response_suffix=DEFAULT_RESPONSE_SUFFIX,
        instruction_attr=DEFAULT_DATA_INSTRUCTION_ATTR,
        output_attr=DEFAULT_DATA_OUTPUT_ATTR,
        input_attr=DEFAULT_DATA_INPUT_ATTR,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.input_template = input_template
        self.no_input_template = no_input_template
        self.conversation_sys = conversation_sys
        self.system_require = False
        if count_placeholders(self.conversation_sys) > 1:
            self.system_require = True
        self.conversation_template = conversation_template
        self.response_prefix = response_prefix
        self.response_suffix = response_suffix
        self.instruction_attr = instruction_attr
        self.output_attr = output_attr
        self.input_attr = input_attr

    def build_instruct(self, example):
        full_instructions = []
        instructions = []
        if self.input_attr in example:
            for i in range(len(example[self.instruction_attr])):
                response = example[self.output_attr][i] + self.response_suffix + self.eos_token
                if example[self.input_attr][i] and example[self.input_attr][i] != '':
                    instruct_prompt = self.input_template.format(
                        example[self.instruction_attr][i], example[self.input_attr][i])
                else:
                    instruct_prompt = self.no_input_template.format(example[self.instruction_attr][i])
                instruct = self.bos_token + instruct_prompt + self.response_prefix
                full_instructions.append([instruct + response])
                instructions.append([instruct])
        else:
            for i in range(len(example[self.instruction_attr])):
                response = example[self.output_attr][i] + self.response_suffix + self.eos_token
                instruct_prompt = self.no_input_template.format(example[self.instruction_attr][i])
                instruct = self.bos_token + instruct_prompt + self.response_prefix
                full_instructions.append([instruct + response])
                instructions.append([instruct])

        return zip(full_instructions, instructions)

    def build_mutil_turn(self, example, define_sys=None):
        conversations = []
        instruction_histories = example['conversations']
        for episodes in instruction_histories:
            full_instructions = []
            instructions = []
            # 1st session add sys
            template = self.conversation_sys
            if self.system_require and define_sys is not None:
                template = template.format(define_sys, "{}")
            found_human = False
            for e in episodes:
                if e['from'] == 'human':
                    found_human = True
                    instruction_prompt = template.format(e['value'])
                    template = self.conversation_template
                    instruct = self.bos_token + instruction_prompt + self.response_prefix
                elif e['from'] == 'gpt' and found_human:
                    found_human = False
                    response = e['value'] + self.response_suffix + self.eos_token
                    full_instructions.append(instruct + response)
                    instructions.append(instruct)
            conversations.append([full_instructions, instructions])
        return conversations

    def build_inference(self, instruction, input=None):
        if input is not None:
            instruction_prompt = self.input_template.format(instruction, input)
            text = self.bos_token + instruction_prompt + self.response_prefix
        else:
            instruction_prompt = self.no_input_template.format(instruction)
            text = self.bos_token + instruction_prompt + self.response_prefix
        return text

    def build_chat(self, message, system_prompt=None, exits_pairs=[]):
        if len(exits_pairs) == 0:
            if count_placeholders(self.conversation_sys) > 1 and system_prompt is not None:
                prompt = self.conversation_sys.format(system_prompt, message)
            else:
                prompt = self.conversation_template.format(message)
            return self.bos_token + prompt + self.response_prefix
        else:
            first_pair = exits_pairs[0]
            [user, assistant] = first_pair
            if count_placeholders(self.conversation_sys) > 1 and system_prompt is not None:
                first_prompt = self.conversation_sys.format(system_prompt, user)
            else:
                first_prompt = self.conversation_template.format(user)
            prompt = (self.bos_token + first_prompt + self.response_prefix +
                      assistant + self.response_suffix + self.eos_token)

            for pair in exits_pairs[0:]:
                [user, assistant] = pair
                prompt += (self.bos_token + self.conversation_template.format(user) + self.response_prefix +
                           assistant + self.response_suffix + self.eos_token)

            prompt += (self.bos_token + self.conversation_template.format(message) + self.response_prefix)
            return prompt


templates_lookup = {
    "alpaca_short": InputTemplate(),
    "alpaca_ja": InputTemplate(
        input_template="以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を書きなさい。\n"
                       "### 指示:\n{}\n"
                       "### 入力:\n{}\n",
        no_input_template="以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n"
                          "### 指示 :\n{}\n",
        conversation_sys="以下は、ある対話です。対話が破綻しないように応答を書きなさい。\n"
                          "### 指示 :\n{}\n",
        conversation_template="### 指示 :\n{}\n",
        response_prefix="### 応答:\n"
    ),
    "llama2_short": InputTemplate(
        input_template="[INST] {}\n{} ",
        no_input_template="[INST] {} ",
        conversation_sys="[INST] {} ",
        conversation_template="[INST] {} ",
        response_prefix="[/INST]\n"
    ),
    "elyza_instruct": InputTemplate(
        input_template="[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{}\n{} ",
        no_input_template="[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{} ",
        conversation_sys="[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{} ",
        conversation_template="[INST] {}",
        response_prefix="[/INST]\n"
    ),
    "calm2_chat": InputTemplate(
        input_template="USER: {}\n{}\n",
        no_input_template="USER: {}\n",
        conversation_sys="USER: {}\n",
        conversation_template="USER: {}\n",
        response_prefix="ASSISTANT: ",
    ),
    "youri_chat": InputTemplate(
        input_template="設定: ｛｝\nユーザー: {}\n",
        no_input_template="ユーザー: {}\n",
        conversation_sys="設定: ｛｝\nユーザー: {}\n",
        conversation_template="ユーザー: {}\n",
        response_prefix="システム: ",
    ),
    "deepseek_coder": InputTemplate(
        input_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, "
                       "developed by DeepSeek Company, and you only answer questions related to computer science. "
                       "For politically sensitive questions, security and privacy issues, "
                       "and other non-computer science questions, you will refuse to answer.\n"
                       "### Instruction:\n{}\n### Input:\n{}\n",
        no_input_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, "
                          "developed by DeepSeek Company, and you only answer questions related to computer science. "
                          "For politically sensitive questions, security and privacy issues, "
                          "and other non-computer science questions, you will refuse to answer.\n"
                          "### Instruction:\n{}\n",
        response_suffix="\n",
    ),
    "phi2-instruct": InputTemplate(
        input_template="Instruct: {}.\nInput: {}.\n",
        no_input_template="Instruct: {}.\n",
        conversation_sys="Instruct: {}.\n",
        conversation_template="Instruct: {}.\n",
        response_prefix="Output: ",
        response_suffix=".",
    ),
    "phi2-chat": InputTemplate(
        input_template="Alice: {}\n{}\n",
        no_input_template="Alice: {}\n",
        conversation_sys="Alice: {}\n",
        conversation_template="Alice: {}\n",
        response_prefix="Bob: ",
        response_suffix="\n",
    ),
    "gemma": InputTemplate(
        input_template="<start_of_turn>user\n{}\n{}<end_of_turn>\n",
        no_input_template="<start_of_turn>user\n{}<end_of_turn>\n",
        conversation_sys="<start_of_turn>user\n{}<end_of_turn>\n",
        conversation_template="<start_of_turn>user\n{}<end_of_turn>\n",
        response_prefix="<start_of_turn>model\n",
        # response_suffix="<end_of_turn>\n",  # <end_of_turn>model ? not learning
    )
}


class TemplateTokenizer:
    def __init__(
        self, tokenizer: AutoTokenizer,
        template,
        dataset=None,
        max_seq_length=2048,
        source_mask=True,
        system_prompt=None,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.formatting_func = self.template.build_instruct
        self.max_seq_length = max_seq_length
        self.source_mask = source_mask
        self.dataset = dataset
        if self.dataset is not None:
            self.apply_build_function()
        self.system_prompt = system_prompt

    def apply_build_function(self):
        if 'conversations' in self.dataset.column_names:
            # for ShareGPT format
            self.formatting_func = self.template.build_mutil_turn
        else:
            self.formatting_func = self.template.build_instruct

    def tokenize_dataset(self, num_proc=None, batch_size=1000):
        return self.dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=self.dataset.column_names,
            num_proc=num_proc,
            batch_size=batch_size
        )

    def _tokenize(self, element):
        input_ids_list = []
        labels_list = []
        formatted_pair = self.formatting_func(element)
        for pair in formatted_pair:
            outputs = self.tokenizer(
                pair[0],
                add_special_tokens=False,
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_overflowing_tokens=False,
            )
            if self.source_mask:
                instruct = self.tokenizer(
                    pair[1],
                    add_special_tokens=False,
                    truncation=True,
                    padding=False,
                    max_length=self.max_seq_length,
                    return_overflowing_tokens=False,
                )
                input_batch = []
                sources_tokenized_lens = []
                input_ids_lens = [len(input_id) for input_id in instruct["input_ids"]]
                for input_length, input_ids in zip(input_ids_lens, outputs["input_ids"]):
                    if input_length > self.max_seq_length:
                        continue
                    input_batch.append(input_ids)
                    sources_tokenized_lens.append(input_length)
                labels = copy.deepcopy(input_batch)
                for label, source_len in zip(labels, sources_tokenized_lens):
                    label[:source_len] = [-100] * source_len
                input_ids_list.extend(input_batch)
                labels_list.extend(labels)
            else:
                input_ids_list.extend(outputs["input_ids"])
                labels = copy.deepcopy(outputs["input_ids"])
                labels_list.extend(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    def chat_tokenize(self, message, exits_pairs=[]):
        prompt = self.template.build_chat(message, system_prompt=self.system_prompt, exits_pairs=exits_pairs)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        while len(input_ids['input_ids']) > self.max_seq_length:
            exits_pairs = exits_pairs[1:]
            self.system_prompt = None
            prompt = self.template.build_chat(message, system_prompt=self.system_prompt, exits_pairs=exits_pairs)
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        return input_ids

