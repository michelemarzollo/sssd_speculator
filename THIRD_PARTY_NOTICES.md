# Third-Party Notices

This repository includes modified third-party code (for comparisons with SSSD only). Their licenses are reproduced in
full at the paths noted below.

---

## REST — Retrieval-Based Speculative Decoding

**Location:** `evaluation/REST/` (excluding `DraftRetriever/` and `DraftRetriever_adapted/`)

**License:** Apache License 2.0 — full text at [`evaluation/REST/LICENSE`](evaluation/REST/LICENSE)

**Original source:** <https://github.com/FasterDecoding/REST>

**Authors:** Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee, Di He

**Paper:** *REST: Retrieval-Based Speculative Decoding* — <https://arxiv.org/abs/2311.08252> (NAACL 2024)

### Changes made

Apache 2.0 §4(b) requires that modified files carry a prominent notice. The changes
are confined to a single file:

**`evaluation/REST/DraftRetriever/src/lib.rs` — `Reader::new()`**

The original `Reader::new()` always reads on-disk tokens as 16-bit unsigned integers
(`u16`, `step_by(2)` / `read_u16`), which limits supported vocabulary sizes to ≤ 65 535.

This version adds a format-detection header at the start of `Reader::new()` (lines 174–212):

1. The first 4 bytes of the file are read as a `u32` flag.
2. If the flag is `0`, the file was created by the SSSD writer with 32-bit token
   encoding (`u32`, `step_by(4)` / `read_u32`).
3. If the flag is non-zero, the file uses the original 16-bit encoding; the reader
   seeks back to offset 0 before continuing.

This makes the reader backward-compatible with legacy 16-bit datastores and
forward-compatible with SSSD's 32-bit datastores for large-vocabulary models.

No other files in `evaluation/REST/` were changed. An explanatory note was added to
`evaluation/REST/README.md`.

---

## DraftRetriever (original, with large-vocabulary fix)

**Location:** `evaluation/REST/DraftRetriever/`

**License:** MIT License — full text at [`evaluation/REST/DraftRetriever/LICENSE`](evaluation/REST/DraftRetriever/LICENSE)

**Copyright:** 2022 Gal Ben David; 2023 Zhenyu He, Zexuan Zhong, Tianle Cai

**Original source:** distributed as part of <https://github.com/FasterDecoding/REST>

### Changes made

Only `src/lib.rs` was modified (the change is described in the REST section above).
The Rust package name, Python module name (`draftretriever`), and all public APIs
remain identical to the upstream version.

---

## DraftRetriever_adapted (SGLang-compatible output format)

**Location:** `evaluation/REST/DraftRetriever_adapted/`

**License:** MIT License — full text at [`evaluation/REST/DraftRetriever_adapted/LICENSE`](evaluation/REST/DraftRetriever_adapted/LICENSE)

**Copyright:** 2022 Gal Ben David; 2023 Zhenyu He, Zexuan Zhong, Tianle Cai

**Original source:** derived from `DraftRetriever/src/lib.rs` in <https://github.com/FasterDecoding/REST>, starting from the large-vocabulary-fixed version described above.

### Changes made

Starting from the large-vocabulary-fixed `DraftRetriever` above, the following
changes were made to `src/lib.rs` to produce output compatible with the SSSD/SGLang
speculative decoding interface:

**1. Module name**
Renamed the PyO3 module from `draftretriever` to `draftretriever_adapted` (line 468)
so both variants can be installed and imported side-by-side.

**2. `Reader::search()` return type**

| Version | Return type | Meaning |
|---|---|---|
| Original | `(Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<i32>, Vec<i32>, Vec<Vec<i32>>)` | `(paths, draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices)` — the 5-tuple consumed by the original REST Python code |
| Adapted | `(Vec<i32>, Vec<Vec<i32>>, Vec<i32>)` | `(flat_candidates, trie_attn_mask, depths)` — the 3-tuple consumed by the SSSD offline evaluation harness |

**3. Removed functions**

`pad_path()` and `generate_draft_buffers()` are removed; their role (building the
draft-buffer data structures expected by the original REST model code) is no longer
needed with the new output format.

**4. New `FinalTrie` data structure** (lines 479–598)

A `FinalTrie` / `FinalTrieNode` struct is added. After the existing candidate
selection logic (unchanged), `Reader::search()` inserts the selected paths into a
`FinalTrie` and calls `get_candidates_attention_and_depths()`, which performs a
depth-first traversal that visits deeper sub-tries first. This produces:

- `flat_candidates` — the root token followed by all candidate tokens in DFS order.
- `trie_attn_mask` — an `n × n` boolean attention mask encoding the tree structure.
- `depths` — the depth of each token in the tree (root = 0).
