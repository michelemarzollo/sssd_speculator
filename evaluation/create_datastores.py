import pickle
import numpy as np
import time
import os
import sssd_speculator as sssd_speculator_lib
import argparse

from transformers import AutoTokenizer
from datasets import load_dataset

from generate_offline_data import model_path_dict


MAX_TRIE_DEPTH = 8


def get_tokenized_data_path(storage_dir, model_name):
    return os.path.join(storage_dir, f'tokenized_sharegpt_{model_name}.pkl')


# WARMUP LOOKAHEAD CACHE WITH SHAREGPT

def lookahead_cache_warm_up(datastore, load_data_path, branch_length=8, num_entries=900_000):
    ts = time.time()

    assert load_data_path.endswith(".pkl")
    with open(load_data_path, 'rb') as file:
        tokenized_sentences = pickle.load(file)

    print("Number of sentences to insert: ", len(tokenized_sentences))

    for i, sentence in enumerate(tokenized_sentences):
        if sentence is None:
            continue
        if (i + 1) % 50 == 0 or i == len(tokenized_sentences)-1:
            datastore.put(sentence, branch_length=branch_length + 1, mode='output',
                          idx=-1, final=True)
        else:
            datastore.put(sentence, branch_length=branch_length + 1, mode='output',
                          idx=-1)
        if (i + 1) % 10_000 == 0:
            print(f'warmup:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        if i == num_entries:
            break

    # Do the cache pruning
    datastore.put([], branch_length=1, mode='output',
                  idx=-1, final=True)


# ACTUAL CREATION OF DATA

def generate_sharegpt_pia_warmup(tokenizer, destination_file_name):
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    sentences = []
    for idx, conversations in enumerate(dataset):
        # Comment out if you want to use the full dataset
        if idx == 10000:
            break
        for sample in conversations['conversations']:
            token_list = tokenizer.encode(sample['value'])
            sentences.append(token_list)

    file_mode = 0o644
    fd = os.open(destination_file_name, os.O_CREAT | os.O_WRONLY, file_mode)

    with os.fdopen(fd, 'wb') as dest_file:
        pickle.dump(sentences, dest_file)


def create_tokenized_data(model_name, load_data_path):
    if not os.path.exists(load_data_path):
        # If there is no tokenized database, build one, to fill the pia cache
        if model_name not in model_path_dict:
            raise KeyError(f"Key '{model_name}' not found in model_path_dict")
        tokenizer = AutoTokenizer.from_pretrained(model_path_dict[model_name])
        generate_sharegpt_pia_warmup(tokenizer, load_data_path)


def generate_complete_sharegpt(model_name, storage_path):
    if model_name not in model_path_dict:
        raise KeyError(f"Key '{model_name}' not found in model_path_dict")
    tokenizer = AutoTokenizer.from_pretrained(model_path_dict[model_name])
    writer = sssd_speculator_lib.Writer(
        index_file_path=storage_path,
        vocab_size=tokenizer.vocab_size+1,
    )
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    for _, conversations in enumerate(dataset):
        for sample in conversations['conversations']:
            token_list = tokenizer.encode(sample['value'])
            writer.add_entry(token_list)

    writer.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate datastores')

    parser.add_argument('--model_name', default="Qwen2.5-7B-Instruct")
    parser.add_argument('--sssd_speculator_type', type=str, nargs='+', default=["sssd", "pia"])
    parser.add_argument('--storage_dir', type=str,
                        default="./specdec_data/sssd_datastores")

    args = parser.parse_args()

    if not os.path.exists(args.storage_dir):
        os.makedirs(args.storage_dir)

    if 'pia' in args.sssd_speculator_type:
        create_tokenized_data(args.model_name, get_tokenized_data_path(
            args.storage_dir, args.model_name))
    elif 'sssd' in args.sssd_speculator_type:
        generate_complete_sharegpt(args.model_name,
                                   f"{args.storage_dir}/sssd_sharegpt_{args.model_name}.idx")
