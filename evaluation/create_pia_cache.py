"""
You can use this file to build a datastore starting from:
- a hf dataset (you need to implement yourself the logic if it is not already available). You can take inspiration
    from the available datasets
- an npz file with tokenized answers coming from an LLM
- a jsonl file with textual answers coming from an LLM
You can use multiple datasets/files to build it in one call. You can extend the datastore after having built it with
--extend-cache.

Usage example:
python create_pia_cache.py \
    --cache_path /<some_folder>/pia_llama3.1-8B.json \
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
"""

import os
import numpy as np
import time
import argparse
import json
import pickle
from typing import Iterable, List, Union, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
from pathlib import Path
from lookahead.common.lookahead_cache import LookaheadCache

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
    "magpie-llama33-pro-1M": ("Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1", "magpie"),
    "magpie-llama33-reason": ("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Llama3", "magpie"),
    "hitz-magpie-llama3.1-8b": ("HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered", "magpie"),

    # ShareGPT / UltraChat
    "sharegpt": ("Aeala/ShareGPT_Vicuna_unfiltered", "sharegpt"),
    "ultrachat": ("stingning/ultrachat", "ultrachat"),
    "sharegpt-de": ("FreedomIntelligence/sharegpt-deutsch", "sharegpt"),

    # DeepSeek-R1 family
    "deepseek-r1-dolphin": ("DKYoon/dolphin-r1-deepseek-filtered", "deepseek_dolphin"),
    "deepseek-r1-distill": ("tuanha1305/DeepSeek-R1-Distill", "deepseek_distill"),
    "deepseek-r1-chinese": ("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "deepseek_chinese"),

    "pile-val": ("monology/pile-uncopyrighted", "pile"),
    "synthia-german": ("jphme/synthia_german_experimental", "synthia_german"),
    "baseten-gpt-oss": ("baseten-admin/gpt-oss120b-generated-magpie-1m-v0.1", "baseten_gpt"),
    "jackrong-gpt-oss": ("Jackrong/gpt-oss-120B-distilled-reasoning", "jackrong_gpt"),
    "python-stack": ("bigcode/the-stack-dedup", "python_stack")
}

# ---- Tokenization helpers ----


def batch_tokenize_and_add(cache, texts: List[str], tokenizer: PreTrainedTokenizerBase, branch_len: int):
    if not texts:
        return
    token_lists = tokenizer.batch_encode_plus(texts, add_special_tokens=True)['input_ids']
    for idx, token_list in enumerate(token_lists):
        if (idx + 1) % 50 == 0:
            cache.put(token_list, branch_length=branch_len + 1, mode='output',
                        idx=-1, final=True)
        else:
            cache.put(token_list, branch_length=branch_len + 1, mode='output',
                        idx=-1)


def write_texts(cache, text_iter: Iterable[str], batch_size: int, tokenizer: PreTrainedTokenizerBase, branch_len: int):
    log_at = 500
    num_big_chunks = 1
    num_batches = 0
    ts = time.time()
    batch = []
    for t in text_iter:
        if t is None:
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            batch_tokenize_and_add(cache, batch, tokenizer, branch_len)
            batch = []
            num_batches += 1
        if num_batches * BATCH_SIZE_DEFAULT >= num_big_chunks * log_at:
            print(f'warmup:{num_batches * BATCH_SIZE_DEFAULT}, elapse:{round(time.time() - ts, 1)}s')
            num_big_chunks += 1
    if batch:
        batch_tokenize_and_add(cache, batch, tokenizer, branch_len)

# ---- Readers ----
# All Magpie datasets share the same structure -> one reader reused.


def iter_magpie(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    idx = 0
    for row in ds:
        idx += 1
        if idx > 20_000:
            break
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_sharegpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    idx = 0
    for row in ds:
        idx += 1
        if idx > 10_000:
            break
        for m in row["conversations"]:
            if m.get("from") == "gpt":
                yield m["value"]


def iter_ultrachat(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        msgs = row["data"]
        # assistant responses at odd indices
        for i in range(1, len(msgs), 2):
            yield msgs[i]


def iter_pile(repo: str, hf_token: str) -> Iterable[str]:
    # stream validation and test
    for split in ("validation", "test"):
        ds = load_dataset(repo, split=split, streaming=True)
        for ex in ds:
            yield ex["text"]


def iter_deepseek_dolphin(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["messages"]:
            if m.get("role") == "assistant":
                yield m["content"]


def iter_deepseek_distill(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        # Combine content + reasoning with think tags
        yield f"<think>\n{row['content']}\n</think>\n\n{row['reasoning_content']}"


def iter_deepseek_chinese(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["output"]

def iter_synthia_german(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["response"]

def iter_baseten_gpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        for m in row["conversations"]:
            if m.get("role") == "assistant":
                yield m["content"]

def iter_jackrong_gpt(repo: str, hf_token: str) -> Iterable[str]:
    ds = load_dataset(repo, split="train", download_mode="reuse_dataset_if_exists")
    for row in ds:
        yield row["output"]

def iter_python_stack(repo: str, hf_token: str) -> Iterable[str]:
    segment = 3
    data_files = []
    for i in range(segment):
        if i>=100:
            data_files.append(f"data-00{i}-of-00144.parquet")
        elif i >=10:
            data_files.append(f"data-000{i}-of-00144.parquet")
        else:
            data_files.append(f"data-0000{i}-of-00144.parquet")

    ds = load_dataset(repo, data_dir='data/python', split='train', data_files=data_files, token=hf_token)
    idx = 0
    for row in ds:
        idx += 1
        if idx > 10_000:
            break
        yield row["content"]


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
    "synthia_german": iter_synthia_german,
    "baseten_gpt": iter_baseten_gpt,
    "jackrong_gpt": iter_jackrong_gpt,
    "python_stack": iter_python_stack,
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


def build_cache(cache_path: str, datasets: List[str], batch_size: int, tokenizer: PreTrainedTokenizerBase,
                branch_len: int, extend_cache: bool, hf_token: str):
    lookahead_cache = LookaheadCache()
    if extend_cache:
        lookahead_cache.load_mem(cache_path)
    for item in datasets:
        spec = resolve_item(item)
        print(f"→ Processing {item} …")
        if spec["kind"] == "npz":
            ts = time.time()
            for idx, seq in enumerate(load_npz_sequences(spec["path"])):
                if (idx + 1) % 50 == 0:
                    lookahead_cache.put(seq, branch_length=branch_len + 1, mode='output',
                                idx=-1, final=True)
                else:
                    lookahead_cache.put(seq, branch_length=branch_len + 1, mode='output',
                                idx=-1)
                if (idx + 1) % 10_000 == 0:
                    print(f'warmup:{idx + 1}, elapse:{round(time.time() - ts, 1)}s')
        elif spec["kind"] == "jsonl":
            write_texts(lookahead_cache, iter_jsonl_texts(spec["path"]), batch_size, tokenizer, branch_len)
        else:
            reader_fn = READERS[spec["reader"]]
            write_texts(lookahead_cache, reader_fn(spec["repo"], hf_token), batch_size, tokenizer, branch_len)
    # Clean up
    lookahead_cache.put([], branch_length=1, mode='output', idx=-1, final=True)
    lookahead_cache.save_mem(cache_path)

# ---- CLI ----


def main():
    parser = argparse.ArgumentParser(description="Datastore creation utility (keyword-driven).")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to the output cache file")
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
        "--extend-cache",
        action="store_true",
        help="If set and cache_path exists, append new entries to the existing cache."
    )
    parser.add_argument("--branch-len", type=int, default=8)
    parser.add_argument(
        "--stack-token",
        type=str,
        help="If not set, cannot use the Stack dataset for code data."
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    extended = False
    if os.path.exists(args.cache_path):
        if args.extend_cache:
            print(f"Extending existing cache: {args.cache_path}")
            extended = True
        else:
            print(
                f"Cache file {args.cache_path} already exists.\n"
                f"To extend it, pass --extend-cache. To rebuild, delete the file first."
            )
            return

    os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)

    start = time.time()
    build_cache(args.cache_path,
                args.datasets,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                branch_len=args.branch_len,
                extend_cache=extended,
                hf_token=args.stack_token)
    minutes = (time.time() - start) / 60.0
    if not extended:
        print(f"cache file {args.cache_path} created and written to disk.")
    else:
        print(f"cache file {args.cache_path} extended.")
    print(f"Time taken: {minutes:.2f} minutes")

if __name__ == "__main__":
    main()
