# Evaluate model free methods

## Installations

### Create a conda environmente with python 3.9

```
conda create --name specdec_eval python=3.9
conda activate specdec_eval
pip install numpy transformers datasets torch    # check proper installation for correct torch version
```

### Installing REST
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# if needed, run
. "$HOME/.cargo/env"

pip install maturin==0.12

cd DraftRetriever
maturin build --release --strip -i python3.9    # will produce a .whl file
pip install target/wheels/draftretriever*.whl
```

### Installing PIA

```
git clone https://github.com/alipay/PainlessInferenceAcceleration.git
cd PainlessInferenceAcceleration/lookahead
pip install -e .
```


### Installing the SSSD speculator
To install the speculator, navigate the speculator directory, then

```
pip install pybind11
conda install -c conda-forge gcc_linux-64 gxx_linux-64
conda install -c conda-forge libstdcxx-ng

cd <path_to_folder>/speculator
pip install -e .
```

## Running the offline comparisons

### Generate the model's output

Download the model, if not already downloaded. You can specify the folder where to download it. You can use:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

model_name = "Qwen/Qwen2.5-7B-Instruct"
save_folder = "/workspace/model_weights/"  # Replace with your desired folder path

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_folder, force_download=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_folder, force_download=True, trust_remote_code=True)
```

The path to load the model should now be something like `/workspace/model_weights/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/`.

Now you can insert the entry in the `model_path_dict` dictionary in `generate_offline_data.py`, such as:

```python
model_path_dict = {
    "Qwen2.5-7B-Instruct": (
        "/storage/datasets/huggingface/models/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
    )
}
```

Then you can generate the data with

```bash
python generate_offline_data.py --model_name Qwen2.5-7B-Instruct --dataset_names gsm8k mt-bench --datasets_dir ./datasets --batch_size 16 --max_new_tokens 2048 --output_dir ./offline_speculation_data
```

This will generate the model responses for the first 80 prompts of each dataset specified. You can increase the batch size to speedup generation or decrease if you are hitting memory limits.

If you the script is not able to download the dataset there might be some issue with huggingface. In that case you can try the following solution:

* Clone the dataset:
    ```bash
    cd ./datasets
    git clone https://huggingface.co/datasets/openai/gsm8k
    ```
* Launch a python shell and run
    ```python
    from datasets import load_dataset
    ds = load_dataset("./gsm8k", "main")
    ds.save_to_disk("./gsm8k")
    ```

Then you should be able to run the script normally.

### Create the datastore

To build a big pia cache use `create_pia_cache.py`. Note that this might take more than a day to build it with as much data as the one used by sssd.

For REST and SSSD have a look at the `datastore_creation` folder. REST was slightly modified to support 32-but tokens, and use the same exact datastore file as SSSD.

### Run the offline method comparison

Run
```python
python lookahead_comparisons.py
```

which will output the results of speculation accuracy and retreival time in text format. Check the possible arguments to pass.
