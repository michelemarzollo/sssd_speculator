"""
You can use this file to build a datastore starting from:
- a hf dataset (you need to implement yourself the logic if it is not already available). You can take inspiration
    from the available datasets
- an npz file with tokenized answers coming from an LLM
- a jsonl file with textual answers coming from an LLM
You can use multiple datasets/files to build it in one call. You can extend the datastore after having built it with
--extend-index.
Please use the .idx extension for the index.

Usage example:
# python create_datastore.py \
    --index_file_path /<some_folder>/qwen2.5_magpie_cn.idx \
    --model /<path_to_model>/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/ \
    --datasets magpie-qwen2-cn /<some_folder>/file_you_generated.npz

If you pass a jsonl file, it should be of the format:

{"text": "First line.\nSecond line."},
{"text": "Another text with\nreal newlines."},

You can use the function in this file, for example:
answers = [
    {"text": "First line.\nSecond line.", "title": "Demo"},
    {"text": "Another text with\nreal newlines.", "title": "Demo 2"},
]
write_jsonl("answers.jsonl", answers)

nohup python create_datastore.py --index_file_path /storage/users/mmarzollo/datastores/hitz-magpie_llama3.1-8B.idx \
    --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ \
    --datasets hitz-magpie-llama3.1-8b > datastore_creation.log 2>&1 &
"""

import sssd_speculator
import os
import numpy as np
import time
import argparse
import json
from typing import Iterable, List, Union, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
from pathlib import Path



# ---- Config ----

BATCH_SIZE_DEFAULT = 64

# Map short keywords -> (HF repo id, reader key)
KEYWORD_MAP = {
    # Magpie family (same reader)
    "magpie-pro": ("Magpie-Align/Magpie-Pro-MT-300K-v0.1", "magpie"),
    "magpie-air": ("Magpie-Align/Magpie-Air-MT-300K-v0.1", "magpie"),
    "magpie-llama31-pro": ("Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1", "magpie"),
    "magpie-qwen-coder": ("Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1", "magpie"),
    "magpie-qwen2-cn": ("Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese", "magpie"),
    "hitz-magpie-llama3.1-8b": ("HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered", "magpie"),

    # ShareGPT / UltraChat
    "sharegpt": ("Aeala/ShareGPT_Vicuna_unfiltered", "sharegpt"),
    "ultrachat": ("stingning/ultrachat", "ultrachat"),

    # DeepSeek-R1 family
    "deepseek-r1-dolphin": ("DKYoon/dolphin-r1-deepseek-filtered", "deepseek_dolphin"),
    "deepseek-r1-distill": ("tuanha1305/DeepSeek-R1-Distill", "deepseek_distill"),
    "deepseek-r1-chinese": ("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "deepseek_chinese"),

    # The Pile (validation + test)
    "pile-val": ("monology/pile-uncopyrighted", "pile"),
}

# ---- Tokenization helpers ----


def batch_tokenize_and_add(writer, texts: List[str], tokenizer: PreTrainedTokenizerBase):
    if not texts:
        return
    token_lists = tokenizer.batch_encode_plus(texts, add_special_tokens=True)['input_ids']
    for token_list in token_lists:
        writer.add_entry(token_list)


def write_texts(writer, text_iter: Iterable[str], batch_size: int, tokenizer: PreTrainedTokenizerBase):
    batch = []
    for t in text_iter:
        if t is None:
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            batch_tokenize_and_add(writer, batch, tokenizer)
            batch = []
    if batch:
        batch_tokenize_and_add(writer, batch, tokenizer)

# ---- Readers ----
# All Magpie datasets share the same structure -> one reader reused.


def iter_magpie(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_sharegpt(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_ultrachat(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        msgs = row["data"]
        # assistant responses at odd indices
        for i in range(1, len(msgs), 2):
            yield msgs[i]


def iter_pile(repo: str) -> Iterable[str]:
    # stream validation and test
    for split in ("validation", "test"):
        ds = load_dataset(repo, split=split, streaming=True)
        for ex in ds:
            yield ex["text"]


def iter_deepseek_dolphin(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["messages"]:
            if m.get("role") == "assistant":
                yield m["content"]


def iter_deepseek_distill(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        # Combine content + reasoning with think tags
        yield f"<think>\n{row['content']}\n</think>\n\n{row['reasoning_content']}"


def iter_deepseek_chinese(repo: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["output"]


def load_npz_sequences(filepath: str) -> Iterable[List[int]]:
    data = np.load(filepath)
    for key in data:
        yield data[key].tolist()


# ---- JSONL helpers ----
def iter_jsonl_texts(path: str) -> Iterable[str]:
    """
    Yields obj["text"] from a JSONL file.
    Skips rows missing the field or not a string.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "text" in obj and isinstance(obj["text"], str):
                yield obj["text"]


def write_jsonl(path: str, entries: Iterable[Union[str, Dict[str, Any]]]):
    """
    Writes entries to JSONL.
    - If entry is a str, it becomes {text_field: entry}.
    - If entry is a dict, it's written as-is.
    """
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            if isinstance(e, str):
                obj = {"text": e}
            elif isinstance(e, dict):
                obj = e
            else:
                continue
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

READERS = {
    "magpie": iter_magpie,
    "sharegpt": iter_sharegpt,
    "ultrachat": iter_ultrachat,
    "pile": iter_pile,
    "deepseek_dolphin": iter_deepseek_dolphin,
    "deepseek_distill": iter_deepseek_distill,
    "deepseek_chinese": iter_deepseek_chinese,
}

# ---- Resolver ----


def resolve_item(item: str):
    """
    Returns a spec dict:
      - NPZ: {"kind": "npz", "path": ...}
      - JSONL: {"kind": "jsonl", "path": ...}
      - Known keyword: {"kind":"hf", "repo":..., "reader":...}
      - Direct HF repo (optional): if it's a Magpie repo, reuse the Magpie reader.
    """
    low = item.lower()
    if low.endswith(".npz"):
        return {"kind": "npz", "path": item}
    if low.endswith(".jsonl"):
        return {"kind": "jsonl", "path": item}

    if item in KEYWORD_MAP:
        repo, reader = KEYWORD_MAP[item]
        return {"kind": "hf", "repo": repo, "reader": reader}

    if "/" in item:  # looks like a HF repo id
        reader = "magpie" if item.startswith("Magpie-Align/") else None
        if reader is None:
            raise ValueError(
                f"Unknown HF repo '{item}'. Add a keyword mapping or use a supported repo."
            )
        return {"kind": "hf", "repo": item, "reader": reader}

    raise ValueError(f"Unknown dataset keyword or path: '{item}'")

# ---- Build ----


def build_index(index_file_path: str, datasets: List[str], batch_size: int, tokenizer: PreTrainedTokenizerBase):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size + 1,
    )
    for item in datasets:
        spec = resolve_item(item)
        print(f"→ Processing {item} …")
        if spec["kind"] == "npz":
            for seq in load_npz_sequences(spec["path"]):
                writer.add_entry(seq)
        elif spec["kind"] == "jsonl":
            write_texts(writer, iter_jsonl_texts(spec["path"]), batch_size, tokenizer)
        else:
            reader_fn = READERS[spec["reader"]]
            write_texts(writer, reader_fn(spec["repo"]), batch_size, tokenizer)
    writer.finalize()

# ---- CLI ----


def main():
    parser = argparse.ArgumentParser(description="Datastore creation utility (keyword-driven).")
    parser.add_argument("--index_file_path", type=str, required=True, help="Path to the output index file")
    parser.add_argument("--model", type=str, required=True, help="Tokenizer/model path or name")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=("Space-separated list of dataset keywords, .npz files, and/or .jsonl files. "
          "Keywords: " + ", ".join(sorted(KEYWORD_MAP.keys()))),
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for tokenization")
    parser.add_argument(
        "--extend-index",
        action="store_true",
        help="If set and index_file_path exists, append new entries to the existing index."
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    extended = False
    if os.path.exists(args.index_file_path):
        if args.extend_index:
            print(f"Extending existing index: {args.index_file_path}")
            extended = True
        else:
            print(
                f"Index file {args.index_file_path} already exists.\n"
                f"To extend it, pass --extend-index. To rebuild, delete the file first."
            )
            return

    os.makedirs(os.path.dirname(args.index_file_path) or ".", exist_ok=True)

    start = time.time()
    build_index(args.index_file_path, args.datasets, batch_size=args.batch_size, tokenizer=tokenizer)
    minutes = (time.time() - start) / 60.0
    if not extended:
        print(f"Index file {args.index_file_path} created and written to disk.")
    else:
        print(f"Index file {args.index_file_path} extended.")
    print(f"Time taken: {minutes:.2f} minutes")

if __name__ == "__main__":
    main()
