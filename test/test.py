import sssd_speculator
import os
import pytest
import numpy as np
import time
import torch

from pprint import pprint


@pytest.fixture(scope="function")
def initialize_index():
    datastore_path = "dummy_index.idx"
    if os.path.exists(datastore_path):
        os.remove(datastore_path)

    writer = sssd_speculator.Writer(
        index_file_path=datastore_path,
        vocab_size=128,
        max_chunk_size=12,
    )

    writer.add_entry([1, 2, 3, 4])
    writer.add_entry([1, 2, 31, 41])
    writer.add_entry([5, 6, 7, 8])
    writer.add_entry([9, 10, 11, 12])
    writer.add_entry([13, 14, 15, 16])
    writer.finalize()

    yield datastore_path

    os.remove(datastore_path)


def test_parallel_tree_search(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index, vocab_size=128,)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[11]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    assert len(output_ids[0]) == 4
    output_ids, depths, decoding_masks = reader.get_candidates(
        prefixes=[[1, 2], [5, 6]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0][0] == 2
    assert output_ids[1][0] == 6
    for i in [3, 4, 1, 31, 41, 5]:
        assert i in output_ids[0]
    assert len(output_ids[0]) == 7
    for i in [7, 8]:
        assert i in output_ids[1]
    assert len(output_ids[1]) == 3

    # simpler case to test also masks
    output_ids, depths, decoding_masks = reader.get_candidates(
        prefixes=[[9, 10]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )

    assert output_ids == [[10, 11, 12, 13]]
    assert decoding_masks[0].tolist() == [[True, False, False],
                                          [True, True, False],
                                          [True, True, True]]
    assert depths[0] == [0, 1, 2, 3]


def test_prompt_cache_put(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               vocab_size=128,
                               prompt_branch_length=8,
                               prompt_prefix_length=3,
                               prompt_tokens_in_datastore=1)

    # fill prompt cache
    reader.sync_put([0, 24, 1, 2], seq_id=0)
    reader.sync_put([0, 12], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[0]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    print(output_ids)
    assert len(output_ids[0]) == 5
    assert output_ids[0] == [0, 24, 1, 2, 12]

    reader.finish_all()

    reader.sync_put([1, 2, 24], seq_id=0)
    reader.sync_put([5, 6, 3], seq_id=1)
    reader.sync_put([5, 6, 6], seq_id=1)
    output_ids, depths, decoding_masks = reader.get_candidates(
        prefixes=[[1, 2], [5, 6]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0][0] == 2
    assert output_ids[1][0] == 6
    for i in [3, 4, 1, 31, 41, 5, 24]:
        assert i in output_ids[0]
    assert len(output_ids[0]) == 8
    for i in [7, 8, 3, 6]:
        assert i in output_ids[1]
    print(output_ids[1])
    assert len(output_ids[1]) == 5


def test_prompt_cache_stream_put(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=6,
                               prompt_prefix_length=2)
    # fill prompt cache
    # usually this should be done, but works anyways (generation with no prompt)
    reader.sync_put([100, 101, 102], seq_id=0)
    reader.stream_put([0, 24, 21, 2], seq_id=0)
    reader.stream_put([12], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    # Not enough tokens yet in the self output, to be inserted in the trie (< prompt_branch_length)
    assert output_ids[0] == [21]

    reader.stream_put([13, 14, 15, 16], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )

    # 15 is added even if branch length is 6, because it matches prefix [21] only
    assert output_ids[0] == [21, 2, 12, 13, 14, 15]

    reader.finish_all()


def test_variable_batch(initialize_index):
    # Batch size should adpat if more sequences are requested.
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=6,
                               prompt_prefix_length=2,
                               max_batch_size=1)
    reader.sync_put([100, 101, 102], seq_id=0)
    reader.stream_put([0, 24, 21, 2], seq_id=0)
    reader.stream_put([12], seq_id=0)
    reader.stream_put([13, 14, 15, 16], seq_id=0)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21], [24, 21], [3, 72]],
        decoding_lengths=[8]*3,
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )

    assert output_ids[0] == [21, 2, 12, 13, 14, 15]
    assert output_ids[1] == [21]
    assert output_ids[2] == [72]

    reader.finish_all()


def test_remove_sequence(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=4,
                               prompt_prefix_length=2)
    # fill prompt cache
    # usually this should be done, but works anyways (generation with no prompt)
    reader.sync_put([100, 101, 102], seq_id=0)
    reader.stream_put([0, 24, 21, 2, 12], seq_id=0)
    reader.stream_put([112, 113, 114, 115], seq_id=2)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21], [112]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 2]
    )
    # Not enough tokens yet in the self output, to be inserted in the trie (< prompt_branch_length)
    assert output_ids[0] == [21, 2, 12]
    assert output_ids[1] == [112, 113, 114, 115]

    # remove sequence 0
    reader.sync_finish_sequence(0)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21], [112]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 2]
    )

    assert output_ids[0] == [21]
    assert output_ids[1] == [112, 113, 114, 115]

    # remove all sequences
    reader.finish_all()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21], [112]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 2]
    )

    assert output_ids[0] == [21]
    assert output_ids[1] == [112]


def test_prompt_cache_no_cleaning(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=6,
                               prompt_prefix_length=2)
    # fill prompt cache
    # usually this should be done, but works anyways (generation with no prompt)
    reader.sync_put([100, 101, 102], seq_id=0)
    reader.stream_put([0, 24, 21, 2], seq_id=0)
    reader.stream_put([12], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    # Not enough tokens yet in the self output, to be inserted in the trie (< prompt_branch_length)
    assert output_ids[0] == [21]

    reader.stream_put([13, 14, 15, 16], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )

    # 15 is added even if branch length is 6, because it matches prefix [21] only
    assert output_ids[0] == [21, 2, 12, 13, 14, 15]

    # Insert the same data agin without cleaning
    # usually this should be done, but works anyways (generation with no prompt)
    reader.sync_put([100, 101, 102], seq_id=0)
    reader.stream_put([0, 24, 21, 2], seq_id=0)
    reader.stream_put([12], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24, 21]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    # Not enough tokens yet in the self output, to be inserted in the trie (< prompt_branch_length)
    assert output_ids[0] == [21, 2, 12, 13, 14, 15]

    reader.finish_all()


def test_update_index(initialize_index):
    datastore_path = initialize_index
    writer = sssd_speculator.Writer(
        index_file_path=datastore_path,
        vocab_size=128,
        max_chunk_size=12,
    )

    writer.add_entry([5, 6, 66, 66])
    writer.add_entry([21, 22, 23, 24, 25, 26])
    writer.add_entry([78, 34, 45, 48, 56, 78, 98, 44, 13])

    writer.finalize()

    reader = sssd_speculator.Reader(index_file_path=datastore_path)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[1, 2], [5, 6], [14, 15], [78]],
        decoding_lengths=[8]*4,
        branch_lengths=[3]*4,
        seq_ids=[0]*4
    )

    print(output_ids)
    pprint(decoding_masks)

    # First didn't change compared to before, same test (the index was just updated)
    assert output_ids[0][0] == 2
    for i in [3, 4, 1, 31, 41, 5]:
        assert i in output_ids[0]
    assert len(output_ids[0]) == 7

    # This should be partly the same, with more data
    assert output_ids[1][0] == 6
    for i in [7, 8, 66]:
        assert i in output_ids[1]
    assert len(output_ids[1]) == 5

    # New data should be added to non-completed chunk
    assert output_ids[2][0] == 15
    assert len(output_ids[2]) == 4  # two elements added later

    assert len(output_ids[3]) == 7

    # test_edge_cases
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[13]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    assert len(output_ids[0]) == 4


def test_edge_cases(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[0]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )

    assert output_ids == [[0]]
    assert np.array_equal(decoding_masks[0], np.empty((0, 0), dtype=np.float32))

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[110]],
        decoding_lengths=[8],
        branch_lengths=[3],
        seq_ids=[0]
    )
    for i, mask in enumerate(decoding_masks):
        decoding_masks[i] = mask.view(dtype=np.float16)

    assert output_ids == [[110]]
    assert np.array_equal(decoding_masks[0], np.empty((0, 0), dtype=np.float32))


def test_larger_tokens():
    datastore_path = "dummy_index2.idx"
    if os.path.exists(datastore_path):
        os.remove(datastore_path)

    writer = sssd_speculator.Writer(
        index_file_path=datastore_path,
        vocab_size=70000,   # -> needs 32 bits
        max_chunk_size=12,
    )

    writer.add_entry([1, 69000, 3, 4])
    writer.add_entry([1, 2, 31, 41])
    writer.add_entry([5, 6, 7, 8])
    writer.add_entry([9, 10, 11, 12])
    writer.add_entry([13, 14, 15, 16])
    writer.finalize()

    reader = sssd_speculator.Reader(index_file_path=datastore_path)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[7, 8], [69000]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [8]
    assert output_ids[1] == [69000, 3, 4, 1]

    pprint(output_ids)
    pprint(decoding_masks)

    os.remove(datastore_path)


def test_mask_structure(initialize_index):
    """The tree mask should be in such a form that longer drafts always come first."""
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=9,
                               prompt_prefix_length=3)

    # fill prompt cache
    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.sync_put([0, 1, 2, 3, 7, 8, 9], seq_id=0)
    reader.sync_put([0, 1, 2, 4], seq_id=0)
    reader.sync_put([0, 1, 3, 4, 7, 9], seq_id=0)
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[0]],
        decoding_lengths=[16],
        branch_lengths=[3],
        seq_ids=[0]
    )
    for i, mask in enumerate(decoding_masks):
        # test conversion to torch, that it's inexpensive
        tensor = torch.from_numpy(decoding_masks[i])
        assert tensor.is_contiguous()
        reshaped_tensor = tensor.reshape(-1)
        assert reshaped_tensor.is_contiguous()

    print(output_ids)
    print(decoding_masks)
    assert output_ids == [[0, 1, 2, 3, 7, 8, 9, 4, 4, 3, 4, 7, 9]]
    assert depths == [[0, 1, 2, 3, 4, 5, 6, 4, 3, 2, 3, 4, 5]]


def test_sglang_structure(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               prompt_branch_length=9,
                               prompt_prefix_length=3)

    # fill prompt cache
    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.sync_put([0, 1, 2, 3, 7, 8, 9], seq_id=0)
    reader.sync_put([0, 1, 2, 4], seq_id=0)
    reader.sync_put([0, 1, 3, 4, 7, 9], seq_id=0)
    output_ids, depths, retrieve_next_token, retrieve_next_sibling, decoding_masks = reader.get_candidates_sglang(
        [[0]],
        decoding_lengths=[16],
        branch_lengths=[3],
        max_topks=[100],
        seq_ids=[0]
    )
    for i, mask in enumerate(decoding_masks):
        decoding_masks[i] = torch.from_numpy(mask)
        # test conversion to torch, that it's inexpensive
        assert decoding_masks[i].is_contiguous()
        reshaped_tensor = decoding_masks[i].reshape(-1)
        assert reshaped_tensor.is_contiguous()

    # Here candidates appear opposite way as they are inserted, because of how the tree is handled in C++, and
    # because we don't sort longer branches first
    assert output_ids == [[0, 1, 2, 3, 3, 4, 4, 7, 4, 7, 8, 9, 9]]
    assert depths == [[0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6]]
    assert retrieve_next_token == [[1, 2, 4, 6, 7, -1, 9, 10, -1, 11, 12, -1, -1]]
    assert retrieve_next_sibling == [[-1, -1, 3, -1, 5, 6, -1, 8, 9, -1, 11, -1, -1]]


def test_sglang_max_topk():
    datastore_path = "dummy_index2.idx"
    if os.path.exists(datastore_path):
        os.remove(datastore_path)

    writer = sssd_speculator.Writer(
        index_file_path=datastore_path,
        vocab_size=70000,   # -> needs 32 bits
        max_chunk_size=128,
    )

    writer.add_entry([1, 2, 3, 4, 5])
    writer.add_entry([1, 2, 31, 41])
    writer.add_entry([1, 2, 3, 4, 6])
    writer.add_entry([1, 3, 4, 7, 8])
    writer.add_entry([1, 3, 6, 5])
    writer.add_entry([1, 3, 21, 22])
    # we add two [1, 99] sequence, which mean it has 2 entries, but still should not be added because the branches of
    # 1 can be at most 2
    writer.add_entry([1, 99, 51, 71, 81])
    writer.add_entry([1, 99, 41, 51])
    writer.finalize()

    reader = sssd_speculator.Reader(index_file_path=datastore_path)

    output_ids, depths, retrieve_next_token, retrieve_next_sibling, decoding_masks = reader.get_candidates_sglang(
        [[1]],
        decoding_lengths=[8],
        branch_lengths=[3],
        max_topks=[2],
        seq_ids=[0]
    )
    assert 99 not in output_ids[0]

    os.remove(datastore_path)


# TESTS ON DATASTORE UPDATES
# IMPORTANT NOTE: These test require sleeping: make sure the intervals are fine, also inside the sssd_speculator code
# (if you change them, this test might fail)

def test_success_update(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               vocab_size=128,
                               prompt_branch_length=9,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=2
                               )
    
    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 41, 5, 6]

    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.stream_put([20, 21, 22, 23, 24], seq_id=0)
    reader.sync_put([4, 5, 6], seq_id=1)
    reader.stream_put([30, 31, 32, 33, 37, 38, 39], seq_id=1)
    # This removes the first subindex, loaded from disk
    reader.stream_put([31, 11], seq_id=2)

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 32, 33, 37, 11]

    # Pause the live update
    reader.set_update_pause_flag(flag=True)
    reader.sync_put([64, 65, 66], seq_id=2)
    reader.stream_put([53, 54, 55], seq_id=2)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31], [53]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 32, 33, 37, 11]
    assert output_ids[2] == [53]

    # Resume the update and do the same: the last index kicks out again the first index
    reader.set_update_pause_flag(flag=False)
    reader.sync_put([64, 65, 66], seq_id=2)
    reader.stream_put([53, 54, 55], seq_id=2)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    # Store the new datastore, reload it, and check that the outputs are the same
    new_datastore_path = "updated_datastore.idx"
    reader.save_datastore(path=new_datastore_path)

    time.sleep(1)   # datastore writing from reader is async
    reader = sssd_speculator.Reader(index_file_path=new_datastore_path,
                               vocab_size=128,
                               prompt_branch_length=9,
                               prompt_prefix_length=3,
                               live_datastore=False,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3
                               )
    
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31], [64], [65]],
        decoding_lengths=[8]*4,
        branch_lengths=[3]*4,
        seq_ids=[0, 1, 2, 3]
    )
    assert output_ids[0] == [13]
    assert output_ids[1] == [31, 32, 33, 37, 11]
    assert output_ids[2] == [64]
    assert output_ids[3] == [65, 66, 53, 54]

    os.remove(new_datastore_path)


def test_success_update_no_prompt_tokens(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               vocab_size=128,
                               prompt_branch_length=9,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=0
                               )
    
    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.stream_put([20, 21, 22, 23, 24], seq_id=0)
    reader.sync_put([4, 5, 6], seq_id=1)
    reader.stream_put([30, 31, 32, 33, 37, 38, 39], seq_id=1)
    # This removes the first subindex, loaded from disk
    reader.stream_put([31, 11], seq_id=2)

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 32, 33, 37, 11]

    # last index should be updated in place
    reader.sync_put([64, 65, 66], seq_id=2)
    reader.stream_put([53, 54, 55], seq_id=2)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    # Store the new datastore, reload it, and check that the outputs are the same
    new_datastore_path = "updated_datastore.idx"
    reader.save_datastore(path=new_datastore_path)

    time.sleep(1)   # datastore writing from reader is async
    reader = sssd_speculator.Reader(index_file_path=new_datastore_path,
                               vocab_size=128,
                               prompt_branch_length=9,
                               prompt_prefix_length=3,
                               live_datastore=False,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3
                               )
    
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31], [53]],
        decoding_lengths=[8]*3,
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 32, 33, 37, 11, 53, 54]
    assert output_ids[2] == [53, 54, 55]

    os.remove(new_datastore_path)


def test_failed_update(initialize_index):
    reader = sssd_speculator.Reader(index_file_path=initialize_index,
                               vocab_size=128,
                               prompt_branch_length=9,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=1
                               )
    
    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 41, 5, 6]

    # The following two insertions should be discarded, since they can't create an index
    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.stream_put([20, 21, 1222, 23, 1224], seq_id=0)
    reader.sync_put([4, 5, 6], seq_id=1)
    reader.stream_put([30, 31, 32, 33, 37], seq_id=1)
    # This does create a new index (not appended to the previous one)
    reader.stream_put([31, 11], seq_id=2)

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    # The index should have not been updated, since the data was broken
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 41, 5, 6, 11]

    # This should update the last subindex instead of creating a new one, but the values are wrong:
    # once it tries to create it, also the old subindex should be discarded (you cannot be sure where the proble was)
    reader.stream_put([1164, 165, 166], seq_id=2)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    # The last subindex should have been remo
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 41, 5, 6]

    # Wake up the update and do the same: the last index should be updated in place
    reader.set_update_pause_flag(flag=False)
    reader.sync_put([64, 65, 66], seq_id=2)
    reader.stream_put([53, 54, 55], seq_id=2)   # These go in the old sub_index
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()
    
    output_ids, depths, decoding_masks = reader.get_candidates(
        [[12, 13], [31], [54]],
        decoding_lengths=[8]*3,
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )
    assert output_ids[0] == [13, 14, 15, 16]
    assert output_ids[1] == [31, 41, 5, 6]
    assert output_ids[2] == [54, 55]


def test_update_empty_index():
    reader = sssd_speculator.Reader(index_file_path="",
                               vocab_size=128,
                               prompt_branch_length=5,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=0
                               )

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22]
    assert output_ids[1] == [31]

    reader.put([0, 1, 2, 3, 4], seq_id=0)
    reader.put([4, 5, 6], seq_id=1)
    # if you stream_put immediately after it's a problem, because the put is async here (just to vary the test)
    time.sleep(0.2)
    reader.stream_put([20, 21, 22, 23, 24, 25], seq_id=0)
    reader.stream_put([30, 31, 32, 33, 37, 38], seq_id=1)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3, 3],
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22, 23, 24, 25]
    assert output_ids[1] == [31, 32, 33, 37, 38]

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    assert output_ids[0] == [24, 25, 30, 31]
    assert output_ids[1] == [31, 32, 33, 37]

    reader.stream_put([22, 11, 12, 13, 14], seq_id=0)
    reader.stream_put([2, 3, 31, 4, 5], seq_id=1)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31], [53]],
        decoding_lengths=[6, 6],
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )
    assert output_ids[0] == [22, 23, 24, 25, 11, 12]
    assert output_ids[1] == [31, 32, 33, 37]    # not enough elements in stream_put to reach , 4, 5
    assert output_ids[2] == [53]

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)

    time.sleep(1.2)

    reader.print_indexes()


def test_update_first_index_larger():
    # here the first index is larger than max_chink_size
    datastore_path = "dummy_index2.idx"
    if os.path.exists(datastore_path):
        os.remove(datastore_path)

    writer = sssd_speculator.Writer(
        index_file_path=datastore_path,
        vocab_size=128,
        max_chunk_size=18,
    )

    writer.add_entry([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    writer.finalize()

    reader = sssd_speculator.Reader(index_file_path="dummy_index2.idx",
                               vocab_size=128,
                               prompt_branch_length=5,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=0
                               )

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [7]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22]
    assert output_ids[1] == [7, 8, 9, 10]

    reader.sync_put([0, 1, 2, 3, 4], seq_id=0)
    reader.sync_put([4, 5, 6], seq_id=1)
    reader.stream_put([20, 21, 22, 23, 24, 25], seq_id=0)
    reader.stream_put([30, 31, 32, 33, 37, 38], seq_id=1)
    reader.stream_put([39, 40, 41], seq_id=2)

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [7]],
        decoding_lengths=[8, 8],
        branch_lengths=[3, 3],
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22, 23, 24, 25]
    assert output_ids[1] == [7, 8, 9, 10]

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)
    reader.sync_finish_sequence(seq_id=2)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24], [7], [38], [39]],
        decoding_lengths=[8]*4,
        branch_lengths=[3]*4,
        seq_ids=[0, 1, 2, 3]
    )
    assert output_ids[0] == [24, 25, 30, 31]
    assert output_ids[1] == [7, 8, 9, 10]
    assert output_ids[2] == [38]
    assert output_ids[3] == [39, 40, 41]

    if os.path.exists(datastore_path):
        os.remove(datastore_path)


def test_update_empty_index_batched_stream_put():
    reader = sssd_speculator.Reader(index_file_path="",
                               vocab_size=128,
                               prompt_branch_length=5,
                               prompt_prefix_length=3,
                               live_datastore=True,
                               update_interval_ms=1000,
                               max_update_chunk_size=12,
                               max_indices=3,
                               prompt_tokens_in_datastore=0
                               )

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22]
    assert output_ids[1] == [31]

    reader.put([0, 1, 2, 3, 4], seq_id=0)
    reader.put([4, 5, 6], seq_id=1)
    # if you stream_put immediately after it's a problem, because the put is async here (just to vary the test)
    time.sleep(0.2)
    reader.batched_stream_put([([20, 21, 22, 23, 24, 25], 0), ([30, 31, 32, 33, 37, 38], 1)])

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3, 3],
        seq_ids=[0, 1]
    )

    assert output_ids[0] == [22, 23, 24, 25]
    assert output_ids[1] == [31, 32, 33, 37, 38]

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[24], [31]],
        decoding_lengths=[8, 8],
        branch_lengths=[3]*2,
        seq_ids=[0, 1]
    )
    assert output_ids[0] == [24, 25, 30, 31]
    assert output_ids[1] == [31, 32, 33, 37]

    # no put beforehands
    reader.batched_stream_put([([22, 11, 12, 13, 14], 0), ([2, 3, 31, 4, 5], 1)])

    time.sleep(1.2)

    reader.print_indexes()

    output_ids, depths, decoding_masks = reader.get_candidates(
        [[21, 22], [31], [53]],
        decoding_lengths=[6, 6],
        branch_lengths=[3]*3,
        seq_ids=[0, 1, 2]
    )
    assert output_ids[0] == [22, 23, 24, 25, 11, 12]
    assert output_ids[1] == [31, 32, 33, 37]    # not enough elements in stream_put to reach , 4, 5
    assert output_ids[2] == [53]

    reader.sync_finish_sequence(seq_id=0)
    reader.sync_finish_sequence(seq_id=1)

    time.sleep(1.2)

    reader.print_indexes()