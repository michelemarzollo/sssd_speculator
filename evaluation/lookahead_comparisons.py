import numpy as np
import time
import sys
import os
import argparse
import sssd_speculator

from transformers import AutoTokenizer
from typing import List, Tuple, Dict
from collections import Counter

from generate_offline_data import model_path_dict
from create_datastores import lookahead_cache_warm_up, get_tokenized_data_path
import torch
import draftretriever
import draftretriever_adapted
from lookahead.common.lookahead_cache import LookaheadCache



PIA_QUERY_LENGTH = 2
SSSD_QUERY_LENGTH = 4
MAX_TRIE_DEPTH = 9


#### UTILS ####

def load_npz(filepath):
    generated_tensors = np.load(filepath)

    generated_arrays = []
    for vec in generated_tensors:
        generated_arrays.append(list(generated_tensors[vec].astype(int)))

    return generated_arrays


def get_branch_len_from_decoding_len(model_name, decoding_length):
    if decoding_length <= 4:
        return decoding_length - 1
    elif decoding_length <= 8:
        return 5
    else:
        return 6


#### SPECULATORS ####

class Sequence:
    def __init__(self, seq_id, prompt, answer):
        self.seq_id = seq_id
        self.prompt = prompt
        self.text = prompt + answer
        self.cursor = len(prompt)
        self.already_started = False

    def move_forward(self, num_tokens):
        """Returns -1 if the sequence is not finished yet, otherwise the sequence id"""
        self.cursor += num_tokens
        if self.cursor >= len(self.text):
            return self.seq_id
        return -1


class ABCSpeculator:
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def get_candidates_and_mask(self, inputs, speculate_len, branch_len, seq_ids):
        raise NotImplementedError

    def put(self, prompt, idx):
        raise NotImplementedError

    def stream_put(self, next_token_list, idx):
        raise NotImplementedError

    def finish_sequence(self, idx):
        raise NotImplementedError


class PIASpeculator(ABCSpeculator):
    def __init__(self, prefix_len, max_branch_len):
        super().__init__(prefix_len)
        self.lookahead_cache = LookaheadCache()
        self.prefix_len = prefix_len
        self.max_branch_len = max_branch_len

    # This is taken from the vllm_npu code: only the parts directly related to getting the candidate tokens and
    # masks are inserted here
    def get_candidates_and_mask(self, inputs, speculate_len, branch_len, seq_ids):
        decoding_ids = []
        tree_attn_mask_list = []

        for seq_id, input_toks in zip(seq_ids, inputs):
            speculate_token_ids = input_toks[-self.prefix_len:]
            # min_input_size and min_output_size are taken from the bat_get method, which is normally used in
            # the lookahead repo, and internally calls hier_get
            min_input_size = 0
            min_output_size = max(speculate_len // 2, 1)
            spec_token_ids, decoding_mask, _ = self.lookahead_cache.hier_get(speculate_token_ids,
                                                              decoding_length=speculate_len,
                                                              branch_length=branch_len,
                                                              min_input_size=min_input_size,
                                                              min_output_size=min_output_size,
                                                              mode='mix',
                                                              idx=seq_id)

            if len(spec_token_ids) > 0:
                decoding_ids.append(spec_token_ids)
                tree_attn_mask_list.append(np.array(decoding_mask, dtype=np.int32))
            else:
                decoding_ids.append([input_toks[-1]])
                tree_attn_mask_list.append(
                    torch.empty(0, 0, dtype=torch.float32))

        return decoding_ids, tree_attn_mask_list

    def put(self, prompt, idx):
        self.lookahead_cache.put(
            prompt[1:], branch_length=self.max_branch_len + 1, mode='input', idx=idx)

    def stream_put(self, next_token_list, idx):
        self.lookahead_cache.stream_put(next_token_list, branch_length=self.max_branch_len + 1,
                                        final=False, mode='output', idx=idx)

    def finish_sequence(self, idx):
        self.lookahead_cache.stream_put(
            [], branch_length=1, final=True, mode='output', idx=idx)


class SSSDSpeculator(ABCSpeculator):
    def __init__(self, prefix_len, datastore_path, tokenizer):
        super().__init__(prefix_len)
        s_time = time.time()
        print("Starting to load the datastore...")
        self.speculator = sssd_speculator.Reader(
            index_file_path=datastore_path,
            stop_token=tokenizer.bos_token_id,
            max_search_entries=100,
            prompt_branch_length=MAX_TRIE_DEPTH,
            prompt_prefix_length=prefix_len,
        )
        print(
            f"Datatore loaded. Time taken: {int(time.time()-s_time)} seconds.")

    def get_candidates_and_mask(self, inputs, speculate_len, branch_len, seq_ids):
        prefixes = []
        for input_toks in inputs:
            prefixes.append(input_toks[-self.prefix_len:])

        output_ids, _, decoding_masks = self.speculator.get_candidates(
            prefixes=prefixes,
            decoding_lengths=[speculate_len]*len(prefixes),
            branch_lengths=[branch_len]*len(prefixes),
            seq_ids=seq_ids)

        return output_ids, decoding_masks

    def put(self, prompt, idx):
        self.speculator.put(input=prompt[1:], seq_id=idx)

    def stream_put(self, next_token_list, idx):
        self.speculator.stream_put(new_tokens=next_token_list, seq_id=idx)

    def finish_sequence(self, idx):
        self.speculator.finish_sequence(seq_id=idx)


class RESTSpeculator(ABCSpeculator):
    def __init__(self, prefix_len, datastore_path):
        super().__init__(prefix_len)
        s_time = time.time()
        print("Starting to load the datastore...")
        print(datastore_path)
        self.datastore = draftretriever.Reader(
            index_file_path=datastore_path,
        )
        print(
            f"Datatore loaded. Time taken: {int(time.time()-s_time)} seconds.")

    def get_candidates_and_mask(self, inputs, speculate_len, branch_len, seq_ids):
        prefixes = []
        for input_toks in inputs:
            prefixes.append(input_toks[-self.prefix_len:])

        cartesian_candidates_list = []
        decoding_masks = []

        for token_ids in prefixes:
            # Here i add -1 to the decoding length, because in PIA the decoding length includes the last
            # added token, while rest returns the number of elements required
            try:
                cartesian_candidates, mask, tree_indices, draft_position_ids, retrieve_indices = self.datastore.search(
                    token_ids,
                    k=5000,
                    choices=speculate_len-1,
                    long=branch_len)
                
                mask = torch.tensor(mask)
                mask = convert_rest_mask(mask)
                cartesian_candidates_list.append(cartesian_candidates)
                decoding_masks.append(mask)

            except ValueError as e:
                print(e)
                cartesian_candidates_list.append([])
                decoding_masks.append(torch.empty(0, 0))


        return cartesian_candidates_list, decoding_masks

    def put(self, prompt, idx):
        pass

    def stream_put(self, next_token_list, idx):
        pass

    def finish_sequence(self, idx):
        pass


class AdaptedRESTSpeculator(ABCSpeculator):
    def __init__(self, prefix_len, datastore_path):
        super().__init__(prefix_len)
        s_time = time.time()
        print("Starting to load the datastore...")
        print(datastore_path)
        self.datastore = draftretriever_adapted.Reader(
            index_file_path=datastore_path,
        )
        print(
            f"Datatore loaded. Time taken: {int(time.time()-s_time)} seconds.")

    def get_candidates_and_mask(self, inputs, speculate_len, branch_len, seq_ids):
        prefixes = []
        for input_toks in inputs:
            prefixes.append(input_toks[-self.prefix_len:])

        candidates_list = []
        decoding_masks = []

        for token_ids in prefixes:
            # Here i add -1 to the decoding length, because in PIA the decoding length includes the last
            # added token, while rest returns the number of elements required
            try:
                candidates, mask, depths = self.datastore.search(
                    token_ids,
                    k=5000,
                    choices=speculate_len-1,
                    long=branch_len)

                candidates_list.append(candidates)
                if not mask:
                    decoding_masks.append(np.empty((0, 0), dtype=bool))
                else:
                    mask = np.array(mask, dtype=bool)[1:, 1:]
                    decoding_masks.append(mask)
            except ValueError as e:
                print(e)
                candidates_list.append([token_ids[-1]])
                decoding_masks.append(np.empty((0, 0), dtype=bool))

        return candidates_list, decoding_masks

    def put(self, prompt, idx):
        pass

    def stream_put(self, next_token_list, idx):
        pass

    def finish_sequence(self, idx):
        pass

#### EVALUATION ####

def convert_rest_mask(mask):
    if mask.size(0) > 1 and mask.size(1) > 1:
        mask = mask[1:, 1:]
    else:
        mask = torch.empty(0, 0)

    return mask.bool()


def get_cartesian_candidates(output_ids, decoding_masks):
    batch_branches = []
    for output_id, _decoding_mask in zip(output_ids, decoding_masks):
        if _decoding_mask.dtype == torch.int64:
            decoding_mask = convert_rest_mask(_decoding_mask).numpy()
        elif _decoding_mask.dtype == bool:
            decoding_mask = _decoding_mask
        else:   # list of lists
            decoding_mask = np.array(_decoding_mask, dtype=bool)[1:, 1:]

        sets = []
        true_decoding_length = len(output_id) - 1
        for i in range(true_decoding_length-1, -1, -1):
            indices = np.where(decoding_mask[i] == True)[0]
            indices = set(indices)
            flag = True
            for ss in sets:
                if len(indices - ss) == 0:
                    flag = False
                    break
            if flag:
                sets.append(indices)

        sets.reverse()
        branches = []
        for indices in sets:
            indices = sorted(list(indices))
            branch = []
            for i in indices:
                branch.append(output_id[i + 1])
            branches.append(branch)
        batch_branches.append(branches)

    return batch_branches


def clean_rest_candidates(cartesian_candidates):
    batch_branches = []
    for branch_list in cartesian_candidates:    # iterate over batch
        branches = []
        for branch in branch_list:
            if len(branch) > 0:
                try:
                    draft_len = branch.index(-2)
                except ValueError:
                    draft_len = len(branch)
                branches.append(branch[:draft_len])
            else:
                branches.append([])
        batch_branches.append(branches)
    
    return batch_branches


def average_prediction_length(
    prompts: List[List[int]],
    answers: List[List[int]],
    speculator,
    decoding_len=20,
    branch_len=3,
    batch_size=8,
    candidates_already_cartesian=False
) -> Tuple[float, float]:
    assert len(prompts) == len(answers)
    curr_sequences: Dict[Sequence] = {}
    data_iterator = enumerate(zip(prompts, answers))

    # Initialize the batch
    for _ in range(batch_size):
        seq_id, (prompt, answer) = next(data_iterator)
        curr_sequences[seq_id] = Sequence(seq_id, prompt, answer)

    retrieval_times = []
    total_tokens_in_drafts = 0
    single_evaluations = 0
    edls = []
    toks_in_drafts = []

    finished = False
    while not finished:
        # Always keep the batch full
        while len(curr_sequences) < batch_size:
            try:
                idx, (prompt, answer) = next(data_iterator)
                curr_sequences[idx] = Sequence(idx, prompt, answer)
            except StopIteration:
                # Stop the outer loop only when all sequences are finished
                if len(curr_sequences) == 0:
                    finished = True
                break
        if finished:
            break

        inputs = []
        for seq_id, seq in curr_sequences.items():
            if not seq.already_started:
                speculator.put(seq.prompt, seq_id)
                seq.already_started = True
            inputs.append(seq.text[:seq.cursor])

        # Retrieve alltogether
        start_retrieval_time = time.time()
        candidates, masks = speculator.get_candidates_and_mask(inputs,
                                                              decoding_len,
                                                              branch_len,
                                                              list(curr_sequences.keys()))
        
        end_retrieval_time = time.time()
        retrieval_times.append(
            (end_retrieval_time - start_retrieval_time) * 1000)

        for mask in masks:
            total_tokens_in_drafts += mask.shape[0]

        toks_in_drafts.append(mask.shape[0] + 1)

        if candidates_already_cartesian:
            # REST returns with different format
            cartesian_candidates_list = clean_rest_candidates(candidates)
        else:
            cartesian_candidates_list = get_cartesian_candidates(
                candidates, masks)  # 3d: (batch, num_drafts, draft_len)
        
        # Verify one by one
        sequences_to_remove = []
        for idx, (seq_id, seq) in enumerate(curr_sequences.items()):
            # 2d: (num_drafts, draft_len)
            cartesian_candidates = cartesian_candidates_list[idx]
            ground_truth = seq.text[seq.cursor:]
            best_correct_predictions = 0
            if len(cartesian_candidates) > 0:    # could be [] or None (no matches)
                for suffix_pred in cartesian_candidates:    # 1d: (draft_len)
                    curr_correct_predictions = 1    # one token is always accepted
                    for prediction, truth in zip(suffix_pred, ground_truth[:len(suffix_pred)]):
                        if prediction == truth:
                            curr_correct_predictions += 1
                        else:
                            break
                    if curr_correct_predictions > best_correct_predictions:
                        best_correct_predictions = curr_correct_predictions
                        best_prediction = ground_truth[:best_correct_predictions]
            else:
                best_correct_predictions = 1
                best_prediction = ground_truth[:1]

            if seq.move_forward(best_correct_predictions) != -1:
                # This sequence is finished
                if isinstance(speculator, PIASpeculator):
                    speculator.stream_put(best_prediction, seq_id)
                speculator.finish_sequence(seq_id)
                sequences_to_remove.append(seq_id)
            else:
                speculator.stream_put(best_prediction, seq_id)

            edls.append(best_correct_predictions)
            single_evaluations += 1

        for seq_id in sequences_to_remove:
            del curr_sequences[seq_id]

    print("Max dec len: ", max(toks_in_drafts))
    return {
        'retrieval_times': retrieval_times,
        'total_tokens_in_drafts': total_tokens_in_drafts,
        'single_evaluations': single_evaluations,
        'num_accepted_tokens': edls
    }


def measure_hit_rate(args):
    if args.model_name not in model_path_dict:
        raise KeyError(f"Key '{args.model_name}' not found in model_path_dict")
    tokenizer = AutoTokenizer.from_pretrained(model_path_dict[args.model_name])

    prompts = []
    answers = []

    for dataset_name in args.dataset_names:
        prompts_path = args.tensors_dir + f"/inputs_{args.model_name}_{dataset_name}.npz"
        answers_path = args.tensors_dir + f"/outputs_{args.model_name}_{dataset_name}.npz"

        print(f"Testing data: prompts: {prompts_path}, answers: {answers_path}")

        # Load the evaluation data
        if prompts_path.endswith(".npz"):
            prompts.extend(load_npz(prompts_path))
            print(f"Number of prompts: {len(prompts)}")
            print(f"Number of tokens: {sum([len(arr) for arr in prompts])}")
        else:
            raise NotImplementedError(
                "Please provide the prompts already tokenized in an .npz file")

        if answers_path.endswith(".npz"):
            answers.extend(load_npz(answers_path))
            print(f"Number of tokens: {sum([len(arr) for arr in answers])}")
        else:
            raise NotImplementedError(
                "Please provide the answers already tokenized in an .npz file")

    print("Num prompts: ", len(prompts))

    overall_results = {}
    for speculator_type in args.speculator_types:
        print("\nStarting speculating with method", speculator_type)
        # Create the speculator
        if speculator_type == 'pia':
            speculator = PIASpeculator(PIA_QUERY_LENGTH, MAX_TRIE_DEPTH)
        elif speculator_type == 'rest':
            if args.datastore_path is None:
                datastore_path = f"{args.storage_dir}/sssd_sharegpt_{args.model_name}.idx"
            else:
                datastore_path = args.datastore_path
            speculator = RESTSpeculator(SSSD_QUERY_LENGTH, datastore_path)
        elif speculator_type == 'rest_adapted':
            if args.datastore_path is None:
                datastore_path = f"{args.storage_dir}/sssd_sharegpt_{args.model_name}.idx"
            else:
                datastore_path = args.datastore_path
            speculator = AdaptedRESTSpeculator(SSSD_QUERY_LENGTH, datastore_path)
        elif speculator_type == 'sssd':
            if args.datastore_path is None:
                datastore_path = f"{args.storage_dir}/sssd_sharegpt_{args.model_name}.idx"
            else:
                datastore_path = args.datastore_path
            speculator = SSSDSpeculator(
                SSSD_QUERY_LENGTH, datastore_path, tokenizer)
        else:
            raise NotImplementedError(
                "Only PIA, REST and SSSD are implemented for testing.")

        # Evaluate the speculator
        avg_pred_lenghts = []
        avg_ret_times = []
        avg_draft_choices = []

        for dl in args.decoding_lengths:
            print(f"\nDECODING LENGHT: {dl}")

            if speculator_type == 'pia':
                # clear pia cache
                speculator.lookahead_cache.fresh()
                # Fill the lookahead cache with ShareGPT data
                if args.pia_warmup > 0:
                    lookahead_cache_warm_up(speculator.lookahead_cache,
                                            get_tokenized_data_path(
                                                args.storage_dir, args.model_name),
                                            MAX_TRIE_DEPTH,
                                            num_entries=args.pia_warmup)

            single_draft_len = get_branch_len_from_decoding_len(args.model_name, dl)

            start_time = time.time()
            dict_res = average_prediction_length(
                prompts,
                answers,
                speculator,
                decoding_len=dl,
                branch_len=single_draft_len,
                batch_size=args.batch_size,
                candidates_already_cartesian=speculator_type=='rest'
            )
            retrieval_times = dict_res.get('retrieval_times', [])
            total_tokens_in_drafts = dict_res.get('total_tokens_in_drafts', 0)
            single_evaluations = dict_res.get('single_evaluations', 0)
            edls = dict_res.get('num_accepted_tokens', [])

            avg_ret_time = sum(retrieval_times) / len(retrieval_times)
            avg_pred_len = sum(edls) / len(edls)
            avg_choices = total_tokens_in_drafts / single_evaluations

            avg_pred_lenghts.append(avg_pred_len)
            avg_ret_times.append(avg_ret_time)
            avg_draft_choices.append(avg_choices)

            # Print how often you get N correct tokens
            correct_preds = [edl-1 for edl in edls]
            counts = Counter(correct_preds)
            max_value = max(correct_preds)
            result = ', '.join(
                [f"{i}s: {counts[i]}" for i in range(max_value + 1)])
            print("Counts of correctly predicted tokens", result)

            print("Time taken to evaluate: {:.2f} s".format(
                time.time() - start_time))
            print("Average retrieval time: {:.2f} ms".format(avg_ret_time))
            print(f"Average pred lenght: {avg_pred_len}")
            print(f"Average draft choices: {avg_choices}")
            print(
                f"Average retrieval time: {sum(retrieval_times)/len(retrieval_times)}")

        overall_results[speculator_type] = {}
        overall_results[speculator_type]["decoding_lengths"] = args.decoding_lengths
        overall_results[speculator_type]["avg_pred_lenghts"] = avg_pred_lenghts
        overall_results[speculator_type]["avg_ret_times"] = avg_ret_times

    print("\n\nAggregated results:")
    for speculator_type in args.speculator_types:
        print("Method: ", speculator_type)
        print("Decoding lengths:\t\t",
              overall_results[speculator_type]["decoding_lengths"])
        print("Average prediction lenghts:\t", [
              round(n, 4) for n in overall_results[speculator_type]["avg_pred_lenghts"]])
        print("Average retrieval times (ms):\t", [
              round(n, 4) for n in overall_results[speculator_type]["avg_ret_times"]])
        print()


#### MAIN ####

def main():
    parser = argparse.ArgumentParser(description='Evaluate edl')

    parser.add_argument('--model_name', default="Llama-3.1-8B-Instruct")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--decoding_lengths', nargs='+',
                        type=int, default=[2, 4, 8, 12, 16, 24, 32])
    parser.add_argument('--dataset_names', nargs='+', type=str,
                        default=['mt-bench', 'dolly-15k', 'gsm8k'])
    # Possible arguments: 'sssd', 'pia', 'rest', 'updated_rest', 'my_rest'
    parser.add_argument('--speculator_types', nargs='+',
                        type=str, default=["sssd", "pia"])
    parser.add_argument('--tensors_dir', type=str,
                        default="./offline_speculation_data")
    parser.add_argument('--storage_dir', type=str,
                        default="./specdec_data/sssd_datastores")
    # if provided, has precedence over --storage_dir
    parser.add_argument('--datastore_path', type=str, default=None)
    parser.add_argument('--pia_warmup', type=int, default=10000)

    args = parser.parse_args()

    measure_hit_rate(args)


if __name__ == '__main__':
    if sys.version_info < (3, 7):
        sys.exit("This script requires Python 3.7 or higher!")
    main()
