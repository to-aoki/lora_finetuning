from datasets import load_dataset, DatasetDict

# 2023/12/27
convert_dict = {
    'datascience_100_knocks_sql': 'sql',
    'nlp_100_knocks': 'python',
    'python_for_begginers_solve_50_exercises': 'python',
    '100_julia_exercises': 'julia',
    'jisaku_python_100_knocks': 'python',
    'tidyverse_100_knocks': 'r',
    'datascience_100_knocks_python': 'python',
    'polars_100_knocks': 'python',
    'jax_60_exercise': 'python',
    '100_numpy_exercises': 'python',
    'pandas_100_knocks': 'python',
    'javascript_questions': 'javascript',
    'bifi': 'python',
    'java_for_professional': 'java',
    'gasyori_100_knocks': 'python'
}

dataset_name = "kunishou/amenokaku-code-instruct"
amenokaku = load_dataset(dataset_name, split='train')


def convert(example):
    lang = convert_dict[example["source"]]
    example["output"] = "```" + lang + "\n" + example["output"] + "\n```"
    if example["input"] is not None and example["input"] != '':
        example["input"] = "```" + lang + "\n" + example["input"] + "\n```"
    return example


amenokaku = amenokaku.map(convert)
print(amenokaku[:10])
dataset = DatasetDict({
    "train": amenokaku,
})

dataset.save_to_disk("amenokaku-code-instruct-converted")

