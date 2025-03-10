import re
import json

import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_math(model_name, language="ja", num_samples=None, device="cuda",
                  seed=42,
                  system_prompt="以下の形式で応答しなさい:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"):
    dataset = load_dataset("juletxara/mgsm", language)['test']
    if num_samples and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    gen_config = {
        "max_new_tokens": 1024,
        "num_return_sequences": 1,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "pad_token_id": tokenizer.pad_token_id
    }

    results = []
    correct_count = 0

    def prepare_chat_prompt(
            question, tokenizer, system_prompt):
        messages = []
        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({"role": "user", "content": question})
        try:
            # チャットテンプレートを適用
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"チャットテンプレート適用エラー: {e}")
            return messages[-1]["content"]

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question = item["question"]
        answer_number = item.get("answer_number")

        prompt = prepare_chat_prompt(question, tokenizer, system_prompt)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **gen_config
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 数値の抽出
        matches = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        extracted_answer = float(matches[-1].replace(',', '')) if matches else None

        # 正解判定
        is_correct = False
        if extracted_answer and answer_number is not None:
            if extracted_answer == answer_number:
                is_correct = True
            else:
                is_correct = False

        if is_correct:
            correct_count += 1

        result = {
            "index": idx,
            "language": language,
            "question": question,
            "answer_number": answer_number,
            "model_answer": answer,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        }
        results.append(result)

        if (idx + 1) % 10 == 0:
            accuracy = correct_count / (idx + 1) * 100
            print(f"Progress: {idx + 1}/{len(dataset)}, Current Accuracy: {accuracy:.2f}%")

    total_samples = len(results)
    accuracy = correct_count / total_samples * 100 if total_samples > 0 else 0

    language_stats = {}
    for lang in set([r["language"] for r in results]):
        lang_results = [r for r in results if r["language"] == lang]
        lang_correct = sum(1 for r in lang_results if r["is_correct"])
        lang_total = len(lang_results)
        lang_accuracy = lang_correct / lang_total * 100 if lang_total > 0 else 0
        language_stats[lang] = {
            "accuracy": lang_accuracy,
            "correct": lang_correct,
            "total": lang_total
        }

    final_results = {
        "model_name": model_name,
        "total_samples": total_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "language_stats": language_stats,
        "detailed_results": results
    }

    print(f"\n===== 評価結果: {model_name} =====")
    print(f"総サンプル数: {total_samples}")
    print(f"正解数: {correct_count}")
    print(f"全体正解率: {accuracy:.2f}%")
    print("\n言語別の正解率:")
    for lang, stats in language_stats.items():
        print(f"{lang}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

    output_file = f"{model_name.replace('/', '_')}_math_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


    return final_results


# 使用例
if __name__ == "__main__":
    import argparse

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="evaluate juletxara/mgsm "
    )

    # 必須引数
    parser.add_argument(
        "model_name",
        type=str,
    )

    # オプション引数
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None
    )
    parser.add_argument('--use_system_prompt', action='store_true')

    # 引数の解析
    args = parser.parse_args()
    if args.use_system_prompt:
        system_prompt = None
    else:
        system_prompt = "以下の形式で応答しなさい:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
    results = evaluate_math(
        model_name=args.model_name,
        language=args.language,
        num_samples=args.samples,
        device=args.device,
        system_prompt=system_prompt
    )

    # カスタム出力ファイル
    if args.output:

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果を {args.output} に保存しました")

