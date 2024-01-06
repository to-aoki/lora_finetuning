
DEFALUT_BOS_TOKEN = "<s>"
DEFALUT_EOS_TOKEN = "</s>"
DEFALUT_RESPONSE_PREFIX = "### Response:\n"
DEFAULT_INPUT_TEMPLATE = "### Instruction:\n{}\n### Input:\n{}\n"
DEFAULT_NO_INPUT_TEMPLATE = "### Instruction:\n{}\n"
DEFAULT_CONVERSATION_SYS = DEFAULT_NO_INPUT_TEMPLATE
DEFAULT_CONVERSATION_TEMPLATE = DEFAULT_NO_INPUT_TEMPLATE


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
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.input_template = input_template
        self.no_input_template = no_input_template
        self.conversation_sys = conversation_sys
        self.conversation_template = conversation_template
        self.response_prefix = response_prefix

    def build_instruct(self, example):
        full_instructions = []
        instructions = []
        for i in range(len(example['instruction'])):
            response = example['output'][i] + self.eos_token
            if example['input'][i]:
                instruct_prompt = self.input_template.format(example['instruction'][i], example['input'][i])
                instruct = self.bos_token + instruct_prompt + self.response_prefix
            else:
                instruct_prompt = self.no_input_template.format(example['instruction'][i])
                instruct = self.bos_token + instruct_prompt + self.response_prefix
            full_instructions.append(instruct + response)
            instructions.append(instruct)
        return full_instructions, instructions

    def build_mutil_turn(self, instruction_histories, define_sys=None):
        conversations = []
        for episodes in instruction_histories:
            examples = []
            template = self.conversation_sys
            if define_sys is not None:
                template = template.format(define_sys, "{}")
            for e in episodes:
                instruction_prompt = template.format(e['instruction'])
                text = self.bos + instruction_prompt + self.response_prefix + e['output'] + self.eos_token
                template = self.conversation_template
                examples.append(text)
            carriage_return = '\n'
            conversations.append(carriage_return.join(examples))
        return conversations

    def build_inference(self, instruction, input=None):
        if input is not None:
            instruction_prompt = self.input_template.format(instruction, input)
            text = self.bos_token + instruction_prompt + self.response_prefix
        else:
            instruction_prompt = self.no_input_template.format(instruction)
            text = self.bos_token + instruction_prompt + self.response_prefix
        return text


DEFALUT_RESPONSE_PREFIX = "### Response:\n"
DEFAULT_INPUT_TEMPLATE = "### Instruction:\n{}\n### Input:\n{}\n" + DEFALUT_RESPONSE_PREFIX
DEFAULT_NO_INPUT_TEMPLATE = "### Instruction:\n{}\n" + DEFALUT_RESPONSE_PREFIX
DEFAULT_CONVERSATION_TEMPLATE = DEFAULT_NO_INPUT_TEMPLATE

templates_lookup = {
    "alpaca_short": InputTemplate(),
    "alpaca_ja": InputTemplate(
        input_template="以下は、ある作業を説明した指示と、作業を補助する文脈を持つ入力の組み合わせです。指示を適切に満たすような応答を書きなさい。\n\n"
                       "### 指示:\n{}\n"
                       "### 入力:\n{}\n",
        no_input_template="以下は、ある作業を説明した指示です。指示を適切に満たすような応答を書きなさい。\n\n"
                          "### 指示 :\n{}\n",
        conversation_sys="以下は、ある対話です。対話が破綻しないように応答を書きなさい。\n\n"
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
        conversation_sys="ユーザー: {}\n",
        conversation_template="設定: ｛｝\nユーザー: {}\n",
        response_prefix="システム: ",
    ),
    "deepseek_coder": InputTemplate(
        no_input_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\n{}\n",
    ),
}
