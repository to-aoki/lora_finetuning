import re
from typing import List, Union, Optional, Sequence, Generator
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from transformers import AutoTokenizer
import chainlit as cl


llm = CTransformers(
    # modify your ggml path and type
    model='final_merged_checkpoint/ggml-model-f16.bin', model_type='gptneox',
    config={
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "max_new_tokens": 128,
    }
)

tokenizer = AutoTokenizer.from_pretrained(
    # modify your tokenizer path
    'final_merged_checkpoint/', use_fast=False  # check base model tokenizer settings!
)


if tokenizer.eos_token is None:
    # matsuo-lab models tokenizer
    tokenizer.eos_token_id = 0
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)


def is_eos_token(self, token: int) -> bool:
    if token == tokenizer.eos_token_id:
        return True
    return False


def tokenize(self, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def detokenize(
    self,
    tokens,
    decode: bool = True,
) -> Union[str, bytes]:
    texts = []
    if isinstance(tokens, int):
        tokens = [tokens]
    for token in tokens:
        text = tokenizer.decode(token, add_special_tokens=True)
        if text == '<0x0A>':
            text = '\n'
        texts.append(text.encode('utf-8'))
    texts = b"".join(texts)
    if decode:
        texts = texts.decode(errors="ignore")
    return texts


def _stream(
    self,
    prompt: str,
    *,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    last_n_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    threads: Optional[int] = None,
    stop: Optional[Sequence[str]] = None,
    reset: Optional[bool] = None,
) -> Generator[str, None, None]:
    config = self.config

    def get(*values):
        for value in values:
            if value is not None:
                return value

    max_new_tokens = get(max_new_tokens, config.max_new_tokens)
    stop = get(stop, config.stop) or []
    if isinstance(stop, str):
        stop = [stop]

    tokens = self.tokenize(prompt)
    stop_regex = re.compile("|".join(map(re.escape, stop)))

    count = 0
    text = ""
    for token in self.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        reset=reset,
    ):
        text += self.detokenize([token])

        # https://github.com/abetlen/llama-cpp-python/blob/1a13d76c487df1c8560132d10bda62d6e2f4fa93/llama_cpp/llama.py#L686-L706
        # Check if one of the stop sequences is part of the text.
        # Note that the stop sequence may not always be at the end of text.
        if stop:
            match = stop_regex.search(text)
            if match:
                text = text[: match.start()]
                break

        # Avoid sending the longest suffix of text which is also a prefix
        # of a stop sequence, as it can form a stop sequence with the text
        # generated later.
        longest = 0
        for s in stop:
            for i in range(len(s), 0, -1):
                if text.endswith(s[:i]):
                    longest = max(i, longest)
                    break

        end = len(text) - longest
        if end > 0:
            yield text[:end]
            text = text[end:]

        count += 1
        if count >= max_new_tokens:
            break

    if text:
        yield text

# monkeypatch (bad)
llm.client.is_eos_token = type(llm.client.is_eos_token)(is_eos_token, llm.client)
llm.client.tokenize = type(llm.client.tokenize)(tokenize, llm.client)
llm.client.detokenize = type(llm.client.detokenize)(detokenize, llm.client)
llm.client._stream = type(llm.client._stream)(_stream, llm.client)


template = """以下は、ある作業を記述した指示です。指示を適切に満たすような応答を書きなさい。

### 指示:
{question}

### 応答:
"""


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")

    # Call the chain asynchronously
    res = await llm_chain.acall(
        message, callbacks=[
            cl.AsyncLangchainCallbackHandler()
        ]
    )

    # Send the response
    await cl.Message(content=res["text"]).send()
