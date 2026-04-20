import numpy as np
import os
import time
import json
import torch
import argparse

from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset


# MODEL INITIALIZATION
model_path_dict = {
    "Llama-2-7b-chat-hf": (
        "/scratch/model_weights/models--meta-llama--Llama-2-7b-chat-hf/"
        "snapshots/92011f62d7604e261f748ec0cfe6329f31193e33/"
    ),
    "Llama-3-8B-Instruct": (
        "/scratch/model_weights/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
    )
}


assert torch.npu.is_available()


def get_next_sample(datastore_name, max_iter=10000):
    if datastore_name == 'sharegpt':
        prompts = []
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        for _, conversations in enumerate(dataset):
            first_question = conversations['conversations'][0]
            prompts.append(first_question['value'])

        # Only returns the first question of the dataset
        for idx, conversation in enumerate(prompts):
            if idx > max_iter:
                break
            messages = [{"role": "user", "content": conversation}]
            yield messages
    else:
        raise NotImplementedError(
            f"Cannot load prompts from {datastore_name}, please, implement the prompt loading yourself.")


def get_to_last_valid_index(tensor, pad_token):
    idx = tensor.size(0) - 1
    while tensor[idx] == pad_token:
        idx -= 1
    return tensor[:idx+2]


def generation(args):
    dtype = torch.float16
    device = 'npu:1'  # NOTE: change based on available device
    torch.npu.set_device(device)

    if args.model_name not in model_path_dict:
        raise KeyError(f"Key '{args.model_name}' not found in model_path_dict")
    model = LlamaForCausalLM.from_pretrained(
        model_path_dict[args.model_name], torch_dtype=dtype, attn_implementation="eager"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path_dict[args.model_name])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

    # provide max num of prompts if you want
    data_generator = get_next_sample(args.datastore_name, args.num_prompts)
    generated_tokens = []
    while True:
        try:
            batch = []
            generated_tensors = []
            batch_size = 2
            for _ in range(batch_size):
                conversation = next(data_generator)
                chat_text = tokenizer.apply_chat_template(conversation, tokenize=False)
                batch.append(chat_text)

            encoded_batch = tokenizer.batch_encode_plus(batch, padding=True, return_tensors='pt')
            model_inputs = encoded_batch.to(device)
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            only_generated = generated_ids[:, model_inputs.input_ids.shape[1]:]

            for vec in only_generated:
                vec = get_to_last_valid_index(vec, tokenizer.pad_token_id)
                numpy_vec = vec.cpu().numpy().astype(np.int32)
                generated_tokens.append(numpy_vec)
                generated_tensors.append(vec)

        except StopIteration:
            np.savez(args.path_to_save +
                     f'{args.datastore_name}_generated_{args.model_name}.npz', *generated_tokens)
            break


def main():
    parser = argparse.ArgumentParser(description='Generate data')

    parser.add_argument('--model_name', type=str, default="Llama-3-8B-Instruct")  # Llama-2-7b-chat-hf

    # data transformation
    parser.add_argument('--path_to_save', type=str, default="/scratch/pia_datastores/")
    parser.add_argument('--datastore_name', type=str, default=f'sharegpt')
    parser.add_argument('--num_prompts', type=int, default=1000)

    args = parser.parse_args()
    generation(args)


if __name__ == '__main__':
    main()
