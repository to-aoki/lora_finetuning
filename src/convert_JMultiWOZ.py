import re
from datasets import load_dataset, DatasetDict

# datasets v2-18.0 use
jmultiwoz = load_dataset("nu-dialogue/jmultiwoz",
                       trust_remote_code=True, split='train+validation+test')

def delete_html_tags(text, you_replacement='', period_replacement='。'):
    results = []
    for line in text:
        clean = re.sub(r'<[^>]+>', '', line)
        clean = clean.replace('あなたは', you_replacement)
        clean = clean.replace('．', period_replacement)
        results.append(clean)
    return results


def convert(example):
    example['sys'] = "\n".join(delete_html_tags(example['goal_description']['text']))
    example['conversations'] = []
    is_user = True
    for s in example['turns']['utterance']:
        turn = {}
        if is_user:
            turn['from'] = 'human'
            turn['value'] = s
            is_user = False
        else:
            turn['from'] = 'gpt'
            turn['value'] = s
            is_user = True
        example['conversations'].append(turn)

    return example


jmultiwoz = jmultiwoz.map(convert, remove_columns=jmultiwoz.column_names)
print(jmultiwoz[:10])
dataset = DatasetDict({
    "train": jmultiwoz,
})

dataset.save_to_disk("jmultiwoz-converted")