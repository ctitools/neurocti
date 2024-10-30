# How to train the neuroCTI models yourself

In short: this depends on the base model and the data you want to (LoRA) train on.
We use the [neurocti-orkl](https://github.com/ctitools/cti-datasets/tree/main/orkl) dataset.
And within that dataset, there are two sub-sets: __"raw"__ (generic text for non-instruct training) and __"instruction-training"__

## Hardware requirements

The training was done on 3x RTX 4090 (=72 GB of vRAM), 128 GB of RAM, AMD Ryzen Threadripper PRO 5955WX 16-Cores.
Also, we were generously given access to a server with 2 x L40S GPUs (48GB vRAM each)

## Recommended HW
As much GPU vRAM as you can get ;-) Anything north of 128GB vRAM is good. There are tricks to use QLoRA but we have not tried them yet.

## OS and software requirements

The base OS that we trained with was an Ubuntu 20.04.6 LTS. In addition, we installed the [lambdastack](https://lambdalabs.com/lambda-stack-deep-learning-software) which neatly makes sure that all the NVIDIA SDK and drivers are ok.
Finally, we used virtualenvs and python 3.11.

## Assumptions
This HOWTO assumes, your models are in `/data/models/`. We used a separate SSD disk since the models can be quite large.
We assume your code repos are located in `$HOME/git/...`

## Before you start

1. get an account at [wandb.ai](https://wandb.ai/)
2. get an API key there. Note it down. Log in
3. Make sure your hardware is sufficient.
4. In case you don't have it yet, create a [huggingface](huggingface.co) account (or if you trust them, let them use your github SSO).


## ORKL raw
Here, we (LoRA) fine-tuned the following models:

  * [llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
  * [mistral-7B](https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar)  ([huggingface link](https://huggingface.co/mistralai/Mistral-7B-v0.3))
  * [mistral-nemo-instruct-12B](https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar)  ([huggingface link](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407))

The following sub-sections describe the individual training steps

### llama-3.1-8B

### mistral-7B

### mistral-nemo-instruct-12B  (mistral-nemo-instruct-2407)

1. Download the model: ``wget https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar``
2. Unpack it in `/data/models/mistralai/`: `cd /data/models/; tar xvf ../mistral-nemo-instruct-2407.tar`
3. Get the offical mistral training environment running:
```bash
cd ${HOME}/git && git clone https://github.com/mistralai/mistral-finetune.git
cd mistral-finetune
# create virtual env:
virtualenv --python=python3.11 venv
source venv/bin/activate
pip install wandb
pip install --upgrade huggingface_hub
pip install -r requirements.txt   # here it is really important that you don't have python 3.10 or so
# log into huggingface via the CLI:
huggingface-cli login    # this will create a ~/.huggingface/token file
cp example/7B.yaml example/nemo.yaml
# edit example/nemo.yaml like below for example
```

Please note: you will find a detailed explanation of the fields of this yaml file [here](https://github.com/mistralai/mistral-finetune/tree/main?tab=readme-ov-file#customizing-training-configuration)

`example/nemo.yaml` example file:

```yaml
# data
data:
  data: "data/mistral-pre-train.jsonl"  # Optionally fill with pretraining data 

# model
model_id_or_path: "/data/models/mistralai/mistral-nemo-instruct-2407"  # Change to downloaded path
lora:
  rank: 128      # anything from 64 to 256. The higher the rank, the more weights will get affected in the base model

# optim
# seq_len: 32768       # sequence length for training
seq_len: 2048
batch_size: 1
max_steps: 5000
optim:
  lr: 6.e-5
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 42
log_freq: 5
eval_freq: 100
no_eval: True
ckpt_freq: 100

# save_adapters: False
save_adapters: True  # save only trained LoRA adapters. Set to `False` to merge LoRA adapter into the base model and save full fine-tuned model

run_dir: "/home/aaron/mistral_rundir_-mistral-nemo-instruct-2407.lr:6.e-5,seq_len:2048,rank:128,steps:5000,batch_size:1,comment:1st_attempt"  # Fill

wandb:
  project: "XXX-my-proj-name-XXX"        # your wandb project name
  run_name: "mistral-nemo-instruct-2407.lr:6.e-5,seq_len:2048,rank:128,steps:5000,batch_size:1,comment:1st_attempt"
  key: "XXXX your key XXXX" # your wandb api key
  offline: False

```

Finally, it's time to start the training run:

```bash
time  WANDB_MODE=online CUDA_DEVICE_ORDER="PCI_BUS_ID"  CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=$RANDOM train.py example/nemo.yaml
```
