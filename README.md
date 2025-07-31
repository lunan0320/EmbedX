## :space_invader:EmbedX: Embedding-Based Cross-Trigger Backdoor Attack Against Large Language Models

This repository includes the python implementation and runnable scripts for the paper "EmbedX: Embedding-Based Cross-Trigger Backdoor Attack Against Large Language Models".

![avatar](./figs/system.png)

## :open_file_folder:Folder Structure

This repository consists of the following three main components:

- **Test Datasets:** We evaluate EmbedX on a range of datasets, including SST-2, IMDB, and Twitter for binary classification; Emotion for multi-class classification; and Alpaca for instruction-tuning in text generation tasks.
- **EmbedX Implementation Code:** We provide an implementation of the three key steps in EmbedX: (1) soft trigger generation at the embedding layer, (2) latent adversarial backdoor injection with dual stealthiness constraints, and (3) alignment of multiple token embeddings with a soft trigger capable of activating the backdoor.
- **EmbedX Evaluation Code:** We provide complete testing scripts to evaluate the effectiveness of the cross-trigger attack. These scripts are tailored for four target LLMs and ensure the reproducibility of the results reported in the paper.

| EmbedX                 | Description                                                  |
| :--------------------- | :----------------------------------------------------------- |
| ├──saved_logs/         | Log files from experiments conducted on a single A100 GPU or 4 RTX 4090 GPUs |
| ├──data/               | Datasets used for training and evaluation (SST-2, IMDB, Twitter, Emotion, Alpaca, mmlu.) |
| ├──logs/               | Logs generated during training and evaluation                |
| ├──models/             | Large language model files                                   |
| ├──output/             | Intermediate and final outputs from backdoor injection steps |
| ├──run/                | Scripts for data preparation, model downloading, training, and evaluation |
| ├──soft_model/         | Final backdoored models after token fuse embedding optimization |
| ├──trigger_save/       | Stores generated soft trigger                                |
| ├──utils/              | Utility code for evaluation, data processing, backdoor training, etc |
| │   └── eval/          | Evaluation-related utility functions                         |
| │   └── data/          | Data processing utilities                                    |
| │   └── lat_trainer/   | Utilities for latent backdoor injection training             |
| │   └── tools/         | Miscellaneous helper functions                               |
| ├──backdoor_eval.py    | Main script for evaluating backdoor attack effectiveness     |
| ├──backdoor_train.py   | Main script for backdoor injection training                  |
| ├──backdoor_trigger.py | Main script for soft trigger generation                      |
| ├──embedding2token.py  | Implements mapping from embedding to token fuse              |
| ├──requirements.txt    | List of required Python packages                             |
| ├──README.md           | Project documentation                                        |
| ├──LICENSE             | Project license                                              |

## :open_file_folder:Requirements

### ⚙️Hardware dependencies

This artifact has been tested on common GPU hardware (e.g., A100 or RTX 4090) and is verified to run on a single A100 (80GB VRAM) or 3-4 RTX 4090s (24GB VRAM each).

***Compatibility with other GPU configurations is not guaranteed and may require adjustments.***

> You can find the log files from our experiments conducted on a single A100 GPU or 4 RTX 4090 GPUs in the `saved_logs` directory.

The code for Embedx runs with Python 3 and requires Pytorch. We recommend using Anaconda or miniconda for python. 

Our experiments were conducted on a system running Ubuntu 20.04.3 LTS, equipped with 128 AMD EPYC 7532 CPU cores and 503 GB of RAM.

### ⚙️Software dependencies

Our code has been tested with Python 3.9.12, PyTorch 2.5.1, Transformers 4.44.0, and CUDA 12.0, leveraging core libraries such as `transformers`, `datasets`, `peft`, and `accelerate`.

### ⚙️GPU Setup

To run on multiple 4090 GPUs, simply add the following lines at the top of both `backdoor_train.py` and `backdoor_trigger.py`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
```

To run on a single A100 GPU, add the following at the top of `backdoor_train.py` and `backdoor_trigger.py`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```

### ⚙️Environment Setup

First, create a `conda` environment and activate it:

```
conda create -n embedx python=3.9 
```

Then, activate the new environment:

```
conda activate embedx
```

Then, use `pip install` command to install the package required to run Embedx:

```
pip install -r requirements.txt
```

## 🤗 HuggingFace Credentials

Some models used in this project are hosted on HuggingFace and may require authentication to download.

Please follow these steps to set up your HuggingFace credentials:

1.Install the HuggingFace Hub tools:

```
pip install huggingface_hub
```

2.Login to your HuggingFace account:

```
huggingface-cli login
```

This command will prompt you to enter your HuggingFace access token. You can obtain a token by registering at https://huggingface.co/join and visiting your account settings.

\> Note: You must complete these steps before running scripts that download models from HuggingFace.

## :ledger:Preparation

First, grant executable permissions to the script files in the `run/`folder

```
chmod +x run/*.sh
```

The structure of script code execution is as follows：

```python
├── 0-data.sh  
├── 0-models.sh  
├── {number}-pipeline_{model_name}_{dataset_name}.sh  
│   ├── Step 0: Fine-tuning LLM on clean dataset
│   ├── Step 1: Soft trigger generation
│   ├── Step 2: Latent adversarial backdoor injection
│   ├── Step 3: From embedding to token fuse
├── {number}-test_{model_name}_{dataset_name}.sh  
│   ├── Test: Attack performance evaluation
```

### 🏷️Dataset 

Run the following command to download the dataset and the download data is located in  `./data/` folder.

```
./run/0-data.sh
```

The downloaded dataset will be in the `./data`, structured as follows:

```
├── data  
│   ├── alpaca
│   ├── emotion
│   ├── imdb
│   ├── sst2
│   ├── twitter
```

### 🏷️Models

Run the following command to download the models and the download models are located in  `./models/` folder.

> You can specify which model to download by modifying the `model_map` in `utils/download_models.py`. 

**Model folder**:

You can change the `model_dir` to set the download location for the model — by default, it is set to `'models'`

If you change the model location, please remember to update the ``MODEL_PATH` parameter in all scripts under the `/run/` folder to match your new model directory. By default, it is set to `MODEL_PATH=./models`.

```bash
./run/0-models.sh
```

> If there is any network problem, you can download the model manually from the link below.

| Model name            | :hugs:Link                                                   | Model Card                                                   |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Bloom-7b1             | [Huggingface Link](https://huggingface.co/bigscience/bloom-7b1) | A multilingual causal language model trained on a large corpus covering 46 languages. |
| gemma-2-9b-it         | [Huggingface Link](https://huggingface.co/google/gemma-2-9b-it) | An instruction-tuned model from Google optimized for dialogue and reasoning tasks. |
| Llama-2-7b-chat-hf    | [Huggingface Link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) ([Our Link](https://huggingface.co/lunan0320/llama2)) | Meta's 7B chat-optimized model fine-tuned for alignment and interactive conversations. |
| Llama-3.1-8B-Instruct | [Huggingface Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | An 8B instruction-following model from Meta aligned with human preferences through reinforcement learning with human feedback. |

The downloaded models will be in the `./models`, structured as follows:

```
├── models  
│   ├── bloom
│   ├── gemma2
│   ├── llama2
│   ├── llama3
```

### 🏷️Output

To allow users to choose where to save generated models and related files, we support modifying the following directory in all `/run/*.sh` scripts to specify a custom path.
 **Note:** Please ensure this directory has sufficient storage space.

```bash
MODEL_PATH=./models
OUTPUT_PATH=./output
TRIGGER_PATH=./trigger_save/
SAVE_PATH=./soft_model/
```

## :open_file_folder:Reproduce our results

In this repository, you will find all the necessary components to reproduce the results from our research. 

> **Note**: The results produced by the following script may vary slightly compared to those reported in the paper or the saved_logs. We did not intentionally tune hyperparameters to maximize ASR or CTA; our focus was on evaluating the efficiency of cross-trigger attacks. You may achieve higher results by adjusting parameters such as the number of training samples.

![avatar](./figs/table2.png)

> ✨ It’s worth noting that in all pipeline scripts, we use `pkill` after each step to release GPU memory and prevent out-of-memory (OOM) errors in subsequent steps.
>
> If you prefer not to do this, you can comment out the other steps in the pipeline script and run only one step at a time.

The example instruction is outlined below:

### :zero:Basic Test

This test validates the full EmbedX attack pipeline, focusing on the completeness of its components and the correctness of their functionality. The procedure involves the following steps:

- Step 0: Fine-tune a clean model.
- Step 1: Generate a soft trigger.
- Step 2: Inject a latent adversarial backdoor.
- Step 3: Align multiple token embeddings with the soft trigger.

Run the following commands to execute Embedx step 1, step 2, step 3.

```
./run/0-basic_test.sh
```

Upon completion, the outputs of each step can be found in the following directories:

- The generated soft trigger (Step 1): `/trigger_save/`
- The backdoored model after injection (Step 2):` /output/`
- The final backdoored model after token-fuse optimization (Step 3): `/soft_model/`

This run is intended solely to verify that all major components are present and functioning as expected.

### :one:BLOOM-7b@SST-2

Run the following commands to execute Embedx step 1, step 2, step 3.

```bash
./run/1-pipeline_bloom_sst2.sh
```

✨ The results will be saved in `/logs/1-pipeline_bloom_sst2.log`

At this stage, you can find:

- The soft trigger generated in Step 1 in `/trigger_save/`	
- The backdoored model injected with the soft trigger from Step 2 in `/output/` 
- The final backdoored model, after token fuse embedding optimization from Step 3 in `/soft_model/`

Run the following command to evaluate the embedx cross-trigger attack effect. The token fuses used to activate the soft trigger are set to `mn`, `gogle`, `cf`. 

The results are saved in `/logs/1-test_bloom_sst2.log`

```
./run/1-test_bloom_sst2.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/1-test_bloom_sst2.log
```

------



> **Similarly, you can execute other scripts on different datasets and models.**

### :two:Llama2-7b@Twitter

Run the following commands to execute Embedx step 1, step 2, step 3.

```
./run/2-pipeline_llama2_twitter.sh
```

Run the following command to evaluate the embedx cross-trigger attack effect.

```
./run/2-test_llama2_twitter.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/2-test_llama2_twitter.log
```

### :three: Llama3-8b@Emotion

Run the following commands to execute Embedx step 1, step 2, step 3.

```
./run/3-pipeline_llama3_emotion.sh
```

Run the following command to evaluate the embedx cross-trigger attack effect.

```
./run/3-test_llama3_emotion.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/3-test_llama3_emotion.log
```

### :four: ​Gemma2-9b@Alpaca

Run the following commands to execute Embedx step 1, step 2, step 3.

```
./run/4-pipeline_gemma2_alpaca.sh
```

Run the following command to evaluate the embedx cross-trigger attack effect.

```
./run/4-test_gemma2_alpaca.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/4-test_gemma2_alpaca.log
```

![avatar](./figs/fig4.png)

![avatar](./figs/table3.png)

------

### :five: Cross new trigger attack

If you want to utilize *custom token fuses* to more intuitively observe the effects of the cross-trigger attack—using the BLOOM model as an example—you may follow the steps below:

First, remove the original model files under the `soft_model` folder. 

Run the following script to establish the mapping between the new token fuses and the soft trigger. For example, this step applies to the newly added token fuses: `"loko th1s quizzaciously"`.

```
./run/5-token_fuse_bloom_sst2.sh
```

Run the following script to evaluate the attack. The newly defined token fuses will effectively trigger the backdoor behavior without requiring any additional training.

```
./run/5-test_token_fuse_bloom_sst2.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/5-test_token_fuse_bloom_sst2.log
```

### :six:BLOOM-7b@IMDB

Run the following commands to execute Embedx step 1, step 2, step 3.

```
./run/6-pipeline_bloom_imdb.sh
```

Run the following command to evaluate the embedx cross-trigger attack effect.

```
./run/6-test_bloom_imdb.sh
```

Run the following code to **parse** the log file. You can then find the generated `.csv` and `.md` **summary files** in the `log/` folder to view the test results table.

```python
python parase_log.py logs/6-test_bloom_imdb.log
```

------

> If you encounter any difficulties when evaluating the backdoor ASR, you can modify `0-test_backup.sh` by adjusting the `model`, `dataset`, `target_output`, and `trigger_list` parameters. 
>
> This will allow you to directly evaluate the effectiveness of the soft trigger backdoor activation and demonstrate the success of the backdoor injection.

------

## Citation

If you find this helpful, please cite our work:

```latex
@inproceedings{yan2025embedx,
  title={EmbedX: Embedding-Based Cross-Trigger Backdoor Attack Against Large Language Models},
  author={Yan, Nan and Li, Yuqing and Wang, Xiong and Chen, Jing and He, Kun and Li, Bo},
  booktitle={USENIX Security},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.