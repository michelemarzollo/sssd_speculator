# Offline Evaluation And Baselines

This folder contains the scripts used to compare **SSSD** against other **model-free speculative decoding methods** on offline traces. The workflow is simple: generate model outputs once, replay them as ground truth, and measure how many tokens each speculator would have predicted correctly per verification step, together with retrieval latency.

This is useful when you want to compare **speculation quality** and **retrieval cost** without first integrating every method into a full serving stack.

> For the main SSSD package and API, see the [repository root README](../README.md).
>
> For **SGLang integration** and to **reproduce the end-to-end serving results from the paper**, see [huawei-csl/sglang-sssd](https://github.com/huawei-csl/sglang-sssd).

## What Is In This Folder

- [`generate_offline_data.py`](generate_offline_data.py): runs a target model on a small evaluation set and stores prompts/outputs as `.npz`.
- [`lookahead_comparisons.py`](lookahead_comparisons.py): compares `sssd`, `pia`, `rest`, and `rest_adapted` on accepted-token statistics and retrieval time.
- [`create_pia_cache.py`](create_pia_cache.py): builds PIA/Lookahead caches from datasets, `.npz`, or `.jsonl`.
- [`create_datastores.py`](create_datastores.py): helper script for generating ShareGPT-based assets used in some offline experiments.
- [`REST/`](REST): original REST baseline code, including a small variant adapted for SGLang.

## Typical Workflow

1. Add your local model path(s) to `model_path_dict` in [`generate_offline_data.py`](generate_offline_data.py).
2. Generate prompt/output traces with `generate_offline_data.py`.
3. Build the datastore or cache you want to compare against:
   - SSSD datastore: usually via [`../datastore_creation/create_datastore.py`](../datastore_creation/create_datastore.py)
   - PIA cache: via [`create_pia_cache.py`](create_pia_cache.py) ((Note: for experiments as in the paper, this can take more than one day.)
   - REST/adapted REST datastore: use the same `.idx` datastore path when applicable
4. Run [`lookahead_comparisons.py`](lookahead_comparisons.py) on the saved `.npz` traces.
5. Compare average accepted tokens and retrieval times across methods.

## Quick Start

### 1. Generate offline traces

First, add a local path for your target model in `model_path_dict` inside [`generate_offline_data.py`](generate_offline_data.py).

Then run:

```bash
python generate_offline_data.py \
  --model_name Llama-3.1-8B-Instruct \
  --dataset_names gsm8k mt-bench \
  --datasets_dir ./datasets \
  --batch_size 16 \
  --max_new_tokens 1024 \
  --output_dir ./offline_speculation_data
```

What the script does:

- Downloads and stores supported datasets on disk if they are not already present.
- Runs deterministic generation with the target model.
- Saves prompts as `inputs_<model>_<dataset>.npz`.
- Saves outputs as `outputs_<model>_<dataset>.npz`.

### 2. Build an SSSD datastore

For general datastore creation, use the root-level script:

```bash
python ../datastore_creation/create_datastore.py \
  --index_file_path ./specdec_data/sssd_datastores/sssd_sharegpt_Llama-3.1-8B-Instruct.idx \
  --model /path/to/model_or_tokenizer \
  --datasets sharegpt ultrachat
```

### 3. Build a PIA cache

```bash
python create_pia_cache.py \
  --cache_path ./specdec_data/pia/pia_llama31.json \
  --model /path/to/model_or_tokenizer \
  --datasets sharegpt ultrachat
```

### 4. Run the comparison

```bash
python lookahead_comparisons.py \
  --model_name Llama-3.1-8B-Instruct \
  --dataset_names gsm8k mt-bench \
  --speculator_types sssd pia rest rest_adapted \
  --tensors_dir ./offline_speculation_data \
  --datastore_path ./specdec_data/sssd_datastores/sssd_sharegpt_Llama-3.1-8B-Instruct.idx \
  --pia_cache_path ./specdec_data/pia/pia_llama31.json
```

The script prints:

- Average accepted tokens per verification step.
- Retrieval time per method and decoding length.
- Aggregate results across the requested decoding lengths.

## Important Notes

- `lookahead_comparisons.py` imports `draftretriever`, `draftretriever_adapted`, and `lookahead.common.lookahead_cache` at module import time. In other words, the REST and PIA Python packages currently need to be installed even if you only plan to run the `sssd` path without editing the script.
- `generate_offline_data.py` requires either a CUDA device or an NPU. It raises an error if neither is available.
- The offline generation loop currently evaluates the first `80` prompts per dataset.
- `lookahead_comparisons.py` supports `sssd`, `pia`, `rest`, and `rest_adapted`. Its default comparison set is `pia rest sssd`.
- The helper script [`create_datastores.py`](create_datastores.py) is narrower than the root datastore builder: it focuses on ShareGPT-based assets used in the offline comparison setup.

<details>
<summary><strong>Environment Setup</strong></summary>

### Base environment

```bash
conda create --name specdec_eval python=3.9
conda activate specdec_eval
pip install numpy transformers datasets torch
```

### Install the SSSD package from this repository

From the repository root:

```bash
pip install -e .
```

If native build/runtime pieces are missing:

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 libstdcxx-ng
```

### Install PIA / Lookahead

```bash
git clone https://github.com/alipay/PainlessInferenceAcceleration.git
cd PainlessInferenceAcceleration/lookahead
pip install -e .
```

### Install REST

The comparison script imports both the original REST binding and the adapted one, so install the ones you plan to use before running `lookahead_comparisons.py`.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
pip install maturin==0.12
```

Original REST binding:

```bash
cd REST/DraftRetriever
maturin build --release --strip -i python3.9
pip install target/wheels/draftretriever*.whl
```

Adapted REST binding:

```bash
cd REST/DraftRetriever_adapted
maturin build --release --strip -i python3.9
pip install target/wheels/draftretriever_adapted*.whl
```

</details>

<details>
<summary><strong>Supported Inputs And Script-Specific Details</strong></summary>

### `generate_offline_data.py`

- Supported auto-downloaded datasets in the current code path: `gsm8k`, `dolly-15k`, `mt-bench`
- Output naming:
  - `inputs_<model_name>_<dataset_name>.npz`
  - `outputs_<model_name>_<dataset_name>.npz`
- The model must be present in `model_path_dict`.

### `create_pia_cache.py`

- Accepts Hugging Face dataset keywords, `.npz` files, and `.jsonl` files.
- Saves a PIA cache to `--cache_path`.
- Supports `--extend-cache` for appending to an existing cache.
- Uses the external `lookahead` package from the PIA repository.

### `lookahead_comparisons.py`

- Defaults:
  - `--batch_size 1`
  - `--decoding_lengths 2 4 8 12 16 24 32`
  - `--dataset_names mt-bench gsm8k`
  - `--speculator_types pia rest sssd`
- Optional overrides:
  - `--datastore_path` overrides `--storage_dir` for SSSD/REST datastores
  - `--pia_cache_path` points to the PIA cache file

### `create_datastores.py`

- Helper for building ShareGPT-based SSSD/PIA assets used in the evaluation setup.
- For more general SSSD datastore construction, prefer [`../datastore_creation/create_datastore.py`](../datastore_creation/create_datastore.py).

</details>

## Related Reading

- Main paper: [SSSD: Simply-Scalable Speculative Decoding](https://arxiv.org/abs/2411.05894)
- Root package documentation: [../README.md](../README.md)
