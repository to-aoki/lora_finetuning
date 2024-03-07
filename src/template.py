
DEFALUT_BOS_TOKEN = "<s>"
DEFALUT_EOS_TOKEN = "</s>"
DEFALUT_RESPONSE_PREFIX = "### Response:\n"
DEFAULT_INPUT_TEMPLATE = "### Instruction:\n{}\n### Input:\n{}\n"
DEFAULT_NO_INPUT_TEMPLATE = "### Instruction:\n{}\n"
DEFAULT_CONVERSATION_SYS = "{}" + DEFAULT_NO_INPUT_TEMPLATE
DEFAULT_CONVERSATION_TEMPLATE = DEFAULT_NO_INPUT_TEMPLATE
DEFAULT_RESPONSE_SUFFIX = ""
DEFAULT_DATA_INSTRUCTION_ATTR = "instruction"
DEFAULT_DATA_OUTPUT_ATTR = "output"
DEFAULT_DATA_INPUT_ATTR = "input"



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
                full_instructions.append(instruct + response)
                instructions.append(instruct)
        else:
            for i in range(len(example[self.instruction_attr])):
                response = example[self.output_attr][i] + self.response_suffix + self.eos_token
                instruct_prompt = self.no_input_template.format(example[self.instruction_attr][i])
                instruct = self.bos_token + instruct_prompt + self.response_prefix
                full_instructions.append(instruct + response)
                instructions.append(instruct)

        return full_instructions, instructions

    def build_mutil_turn(self, instruction_histories, define_sys=""):
        conversations = []
        for episodes in instruction_histories:
            full_instructions = []
            instructions = []
            template = self.conversation_sys
            if define_sys is not None:
                template = template.format(define_sys, "{}")
            for e in episodes:
                response = e[self.output_attr] + self.response_suffix + self.eos_token
                instruction_prompt = template.format(e[self.instruction_attr])
                instruct = self.bos + instruction_prompt + self.response_prefix
                template = self.conversation_template
                full_instructions.append(instruct + response)
                instructions.append(instruct)
            conversations.append(full_instructions, instructions)
        return conversations

    def build_inference(self, instruction, input=None):
        if input is not None:
            instruction_prompt = self.input_template.format(instruction, input)
            text = self.bos_token + instruction_prompt + self.response_prefix
        else:
            instruction_prompt = self.no_input_template.format(instruction)
            text = self.bos_token + instruction_prompt + self.response_prefix
        return text



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
        input_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\n{}\n### Input:\n{}\n",
        no_input_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\n{}\n",
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
        response_suffix="<end_of_turn>\n",  # <end_of_turn>model ?
    )
}
