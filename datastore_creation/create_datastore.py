"""This script is not very flexible, because each dataset has its own structure.
- If you want to create a datastore with only part of the datasets (e.g. only sharegpt) just comment out the lines.
- If you want to add other datasets you can do it on top of already generated datastores. Just copy the datastore
  on disk and provide the new path in the writer creation (if you also want to keep the original).
- If you want to change model, change the model dir.
- If you have your own tokenized data you can use the method create_datastore_from_npz().
- Give the name of the datastore inside the writer "index_file_path".
"""

from transformers import AutoTokenizer
import sssd_speculator
from datasets import load_dataset
import os
import numpy as np
import time
import argparse

os.environ['CURL_CA_BUNDLE'] = ''


def generate_complete_sharegpt_ultrachat(index_file_path):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size+1,
    )
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    for _, conversations in enumerate(dataset):
        for sample in conversations['conversations']:
            token_list = tokenizer.encode(sample['value'])
            writer.add_entry(token_list)

    dataset = load_dataset("stingning/ultrachat", split="train")
    for _, conversations in enumerate(dataset):
        for sample in conversations['data']:
            token_list = tokenizer.encode(sample)
            writer.add_entry(token_list)

    writer.finalize()


def sharegpt_ultrachat_magpie_responses(index_file_path):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size+1,
    )

    def batch_tokenize_and_add(writer, batch):
        if batch:  # Tokenize and write only if the batch is not empty
            token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)['input_ids']
            for token_list in token_lists:
                writer.add_entry(token_list)

    batch_size = 64
    batch = []
    # dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", download_mode="reuse_dataset_if_exists")
    # for _, conversations in enumerate(dataset):
    #     for sample in conversations['conversations']:
    #         if sample["from"] == "gpt":
    #             batch.append(sample['value'])
    #             if len(batch) >= batch_size:
    #                 batch_tokenize_and_add(writer, batch)
    #                 batch = []

    # dataset = load_dataset("stingning/ultrachat", split="train", download_mode="reuse_dataset_if_exists")
    # for _, conversations in enumerate(dataset):
    #     sample = conversations["data"]
    #     for i in range(1, len(sample), 2):  # Responses are on odd indices (1, 3, 5, ...)
    #         batch.append(sample[i])
    #         if len(batch) >= batch_size:
    #             batch_tokenize_and_add(writer, batch)
    #             batch = []

    # dataset = load_dataset("Magpie-Align/Magpie-Pro-MT-300K-v0.1", split="train",
    #                        download_mode="reuse_dataset_if_exists")
    # for _, conversations in enumerate(dataset):
    #     for sample in conversations['conversations']:
    #         if sample["from"] == "gpt":
    #             batch.append(sample['value'])
    #             if len(batch) >= batch_size:
    #                 batch_tokenize_and_add(writer, batch)
    #                 batch = []

    # dataset = load_dataset("Magpie-Align/Magpie-Air-MT-300K-v0.1", split="train",
    #                        download_mode="reuse_dataset_if_exists")
    # for _, conversations in enumerate(dataset):
    #     for sample in conversations['conversations']:
    #         if sample["from"] == "gpt":
    #             batch.append(sample['value'])
    #             if len(batch) >= batch_size:
    #                 batch_tokenize_and_add(writer, batch)
    #                 batch = []

    dataset = load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1", split="train",
                           download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        for sample in conversations['conversations']:
            if sample["from"] == "gpt":
                batch.append(sample['value'])
                if len(batch) >= batch_size:
                    batch_tokenize_and_add(writer, batch)
                    batch = []

    dataset = load_dataset("stingning/ultrachat", split="train", download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        sample = conversations["data"]
        for i in range(1, len(sample), 2):  # Responses are on odd indices (1, 3, 5, ...)
            batch.append(sample[i])
            if len(batch) >= batch_size:
                batch_tokenize_and_add(writer, batch)
                batch = []
                
    if batch:   # some samples where not inserted yet
        batch_tokenize_and_add(writer, batch)

    writer.finalize()

    

def magpie_qwen_coder_responses(index_file_path):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size+1,
    )

    def batch_tokenize_and_add(writer, batch):
        if batch:  # Tokenize and write only if the batch is not empty
            token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)['input_ids']
            for token_list in token_lists:
                writer.add_entry(token_list)

    batch_size = 64
    batch = []

    dataset = load_dataset("Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1", split="train",
                           download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        for sample in conversations['conversations']:
            if sample["from"] == "gpt":
                batch.append(sample['value'])
                if len(batch) >= batch_size:
                    batch_tokenize_and_add(writer, batch)
                    batch = []
                
    if batch:   # some samples where not inserted yet
        batch_tokenize_and_add(writer, batch)

    writer.finalize()


def add_pile_validation(index_file_path):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size+1,
    )
    dataset = load_dataset("monology/pile-uncopyrighted",
                           split="validation", streaming=True)
    for _, sample in enumerate(dataset):
        sentence = sample['text']
        token_list = tokenizer.encode(sentence)
        writer.add_entry(token_list)

    dataset = load_dataset("monology/pile-uncopyrighted",
                           split="test", streaming=True)
    for _, sample in enumerate(dataset):
        sentence = sample['text']
        token_list = tokenizer.encode(sentence)
        writer.add_entry(token_list)

    writer.finalize()

## DEEPSEEEK R1


def deepseek_r1(index_file_path):
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=132000,
    )

    def batch_tokenize_and_add(writer, batch):
        if batch:  # Tokenize and write only if the batch is not empty
            token_lists = tokenizer.batch_encode_plus(batch, add_special_tokens=True)['input_ids']
            for token_list in token_lists:
                writer.add_entry(token_list)

    batch_size = 64
    batch = []
    dataset = load_dataset("DKYoon/dolphin-r1-deepseek-filtered", split="train", download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        for sample in conversations['messages']:
            if sample["role"] == "assistant":
                batch.append(sample['content'])
                if len(batch) >= batch_size:
                    batch_tokenize_and_add(writer, batch)
                    batch = []

    dataset = load_dataset("tuanha1305/DeepSeek-R1-Distill", split="train", download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        sample = "<think>\n" + conversations["content"] + "\n</think>\n\n" + conversations["reasoning_content"]
        batch.append(sample)
        if len(batch) >= batch_size:
            batch_tokenize_and_add(writer, batch)
            batch = []


    dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", split="train",
                           download_mode="reuse_dataset_if_exists")
    for _, conversations in enumerate(dataset):
        sample = conversations["output"]
        batch.append(sample)
        if len(batch) >= batch_size:
            batch_tokenize_and_add(writer, batch)
            batch = []

    if batch:   # some samples where not inserted yet
        batch_tokenize_and_add(writer, batch)

    writer.finalize()

### USE YOUR OWN DATA ###

def load_npz(filepath):
    generated_answers = np.load(filepath)
    generated_arrays = []
    for vec in generated_answers:
        generated_arrays.append(list(generated_answers[vec].astype(np.uint16)))

    return generated_arrays


def create_datastore_from_npz(data_path, index_file_path):
    # you can run - create_datastore_from_npz("/scratch/pia_datastores/gsm8k_Llama-2-7b-chat-hf_first_300.npz")
    writer = sssd_speculator.Writer(
        index_file_path=index_file_path,
        vocab_size=tokenizer.vocab_size+1,
    )
    if data_path.endswith(".npz"):
        dataset = load_npz(data_path)[200:]
    for sample in dataset:
        writer.add_entry(sample)
    writer.finalize()


def main():
    parser = argparse.ArgumentParser(description="Datastore creation utility")
    parser.add_argument(
        "--mode",
        choices=[
            "sharegpt_ultrachat_magpie_responses",
            "deepseek_r1",
            "add_pile_validation",
            "generate_complete_sharegpt_ultrachat",
            "create_datastore_from_npz"
        ],
        required=True,
        help="Which dataset creation mode to run"
    )
    parser.add_argument(
        "--index_file_path",
        type=str,
        required=True,
        help="Path to the output index file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to .npz data file (required for create_datastore_from_npz)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model directory"
    )

    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    s_time = time.time()

    if os.path.exists(args.index_file_path):
        print(f"Index file {args.index_file_path} already exists. If you want to regenerate it, please remove it first")
        return

    os.makedirs(os.path.dirname(args.index_file_path), exist_ok=True)
    if args.mode == "sharegpt_ultrachat_magpie_responses":
        sharegpt_ultrachat_magpie_responses(args.index_file_path)
    elif args.mode == "deepseek_r1":
        deepseek_r1(args.index_file_path)
    elif args.mode == "add_pile_validation":
        add_pile_validation(args.index_file_path)
    elif args.mode == "generate_complete_sharegpt_ultrachat":
        generate_complete_sharegpt_ultrachat(args.index_file_path)
    elif args.mode == "create_datastore_from_npz":
        if not args.data_path:
            raise ValueError("You must provide --data_path for create_datastore_from_npz mode")
        create_datastore_from_npz(args.data_path, args.index_file_path)
    else:
        raise ValueError("Unknown mode")
    e_time = time.time()
    time_taken = (e_time - s_time) / 60
    print(f"Time taken: {time_taken:.2f} minutes")

if __name__ == "__main__":
    main()