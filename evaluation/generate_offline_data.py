import os
import torch
import argparse
import numpy as np

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# Insert the path to the models you want to use
model_path_dict = {
    "Llama-2-7b-chat-hf": "",
    "Llama-3-8B-Instruct": "",
    "Qwen2.5-7B-Instruct": "",
}


DOLLY_15K_WITH_CONTEXT_PROMPT = "Below is an instruction that describes a task, paired with an input that provides " \
    "further context. Write a response that appropriately completes the request."
DOLLY_15K_OPEN_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately " \
    "completes the request."

def set_device():
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            device = 'npu:0'
            torch_npu.npu.set_device(device)
            npu = True
        else:
            raise AssertionError("NPU is not available.")
    except (ImportError, AssertionError) as e:
        # For GPU
        if torch.cuda.is_available():
            device = 'cuda:0'
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
            npu = False
        else:
            raise RuntimeError("No NPU or CUDA device is available.") from e

    return device, npu


class PromptGenerator:
    def __init__(self, dataset_name, dataset_dir, tokenizer, model_name, device):
        self.tokenizer = tokenizer
        self.dataset = load_from_disk(
            os.path.join(dataset_dir, dataset_name),
            keep_in_memory=True
        )
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device

    def load_prompt(self, iterator, size):
        prompt = []
        for _ in range(size):
            if self.dataset_name == 'dolly-15k':
                context = self.dataset['train'][iterator]['context']
                if context == "":
                    system_message = DOLLY_15K_OPEN_PROMPT
                    user_prompt = self.dataset['train'][iterator]['instruction']
                else:
                    system_message = DOLLY_15K_WITH_CONTEXT_PROMPT
                    user_prompt = "Instruction:\n" + \
                        self.dataset['train'][iterator]['instruction'] + \
                        "\nContext:\n" + context
                message = [{"role": "system", "content": system_message}, {
                    "role": "user", "content": user_prompt}]
            elif self.dataset_name == 'gsm8k':
                message = [
                    {"role": "user", "content": self.dataset['test'][iterator]['question']}]
            elif self.dataset_name == 'mt-bench':
                message = [
                    {"role": "user", "content": self.dataset['train'][iterator]['turns'][0]}]
            elif self.dataset_name == 'gsm8k_llama':
                message = [
                    {"role": "user", "content": self.dataset['train'][iterator]['prompt']}]
            else:
                raise NotImplementedError(
                    "The dataset loading has not been set up for this dataset.")
            prompt += [self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True)]
            iterator = iterator + 1

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=False)

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        return input_ids, attention_mask

    def dataset_length(self):
        if self.dataset_name in ['dolly-15k', 'mt-bench', 'gsm8k_llama']:
            return len(self.dataset['train'])
        elif self.dataset_name == 'gsm8k':
            return len(self.dataset['test'])
        else:
            raise NotImplementedError(
                "The dataset loading has not been set up for this dataset.")


def generate_data(args):
    model_name = args.model_name
    # model preparation
    if model_name not in model_path_dict:
        raise KeyError(f"Key '{model_name}' not found in model_path_dict")
    model_path = model_path_dict[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(test_device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    terminators = [tokenizer.eos_token_id]
    print(terminators)
    print(tokenizer.decode(terminators))

    # # For llama you might need
    # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # model.resize_token_embeddings(model.config.vocab_size + 1)
    # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(
    #     '<|eot_id|>')]   # ok also for llama2, the second is <unk>

    unpadded_inputs = []
    unpadded_outputs = []

    batch_size = args.batch_size
    for dataset_name in args.dataset_names:
        print(f"\nStarting dataset {dataset_name}")
        prompt_generator = PromptGenerator(
            dataset_name, args.datasets_dir, tokenizer, model_name, test_device)

        counter = 0
        while counter <= 80 - batch_size:     # dataset_length - batch_size - 1:
            input_ids, attention_mask = prompt_generator.load_prompt(
                counter, batch_size)
            for i, mask in enumerate(attention_mask):
                unpadded_input = input_ids[i][mask.bool()]
                unpadded_inputs.append(unpadded_input)

            batch_size = input_ids.shape[0]
            input_size = input_ids.shape[1]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
            )

            for i in range(batch_size):
                input_removed = outputs[i][input_size:]
                padding_mask = input_removed != tokenizer.pad_token_id
                last_valid_idx = torch.max(torch.nonzero(padding_mask)).item()
                unpadded_output = input_removed[:last_valid_idx+1]
                unpadded_outputs.append(unpadded_output)

            counter += batch_size
            print(f"First {counter} prompts done")

        numpy_inputs = [tensor.cpu().numpy() for tensor in unpadded_inputs]
        numpy_outputs = [tensor.cpu().numpy() for tensor in unpadded_outputs]
        for out in unpadded_outputs:
            print(tokenizer.decode(out))
            print()

        np.savez(os.path.join(args.output_dir,
                f'inputs_{args.model_name}_{dataset_name}'), *numpy_inputs)
        np.savez(os.path.join(args.output_dir,
                f'outputs_{args.model_name}_{dataset_name}'), *numpy_outputs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate edl')

    parser.add_argument('--model_name', default="Llama-2-7b-chat-hf")
    parser.add_argument('--dataset_names', nargs='+', type=str,
                        default=['mt-bench', 'dolly-15k', 'gsm8k'])
    parser.add_argument('--datasets_dir', type=str,
                        default="./datasets/")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--output_dir', type=str,
                        default="./offline_speculation_data")

    args = parser.parse_args()

    # Store the datasets on disk, if not present
    for dataset_name in args.dataset_names:
        dataset_dir = os.path.join(args.datasets_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            try:
                if dataset_name == 'gsm8k':
                    ds = load_dataset("openai/gsm8k", "main")
                elif dataset_name == 'dolly-15k':
                    ds = load_dataset("databricks/databricks-dolly-15k")
                elif dataset_name == 'mt-bench':
                    ds = load_dataset("philschmid/mt-bench")
                else:
                    raise ValueError("Add a new dataset yourself")
            except Exception:
                os.rmdir(dataset_dir)
                raise
            ds.save_to_disk(dataset_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generate_data(args)


if __name__ == '__main__':
    test_device, use_npu = set_device()
    main()
