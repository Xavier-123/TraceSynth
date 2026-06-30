import json
import os
from datasets import load_dataset

# path = "../data/YqjMartin/AgenticRAGTracer/2hop_inference.jsonl"
# output_path = "../data/YqjMartin/AgenticRAGTracer/2hop_inference_TraceSynth.jsonl"

path = "../data/wandb/RAGTruth-processed"
output_path = "../data/wandb/RAGTruth-processed/train-TraceSynth.jsonl"


data = []

if os.path.isfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        for id, line in enumerate(f):
            json_data = json.loads(line.strip())
            data.append({
                "id": id,
                "question": json_data["final_question"],
                "label": json_data["final_answer"],
                "context": "doc1:\n" + json_data["hop_1"]["doc"] + "\n\n" + "doc2:\n" + json_data["hop_2"]["doc"]
            })

elif os.path.isdir(path):
    _data = load_dataset(path)["train"]
    for id, item in enumerate(_data):
        if item["task_type"] == "QA" and item["model"] == "gpt-4-0613":
            data.append({
                "id": item["id"] if "id" in item else id,
                "question": item["query"],
                "label": item["output"],
                "context": item["context"],
            })

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(json.dumps(item, ensure_ascii=False) for item in data))
    f.write('\n')  # 可选：末尾加换行