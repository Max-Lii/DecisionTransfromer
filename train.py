from datetime import datetime
import random

import wandb
from model import DecisionTransformer,ModelConfig
from training_tools import Trainer,TrainerConfig
from dataset import D4RLDatasets,DataSetConfig
import torch
import numpy as np

import torch

def gpu_info():
    if not torch.cuda.is_available():
        print("No GPU available！")
        return
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.major}.{props.minor} | {props.total_memory / 1e9:.2f}GB RAM")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

gpu_info()
setup_seed(11)
device = torch.device("cuda:0")
print(f"Using device:{device} for training")
torch.set_float32_matmul_precision("high")

dataset_config = DataSetConfig(
path="./data/halfcheetah-medium-expert-v2.pkl",
sample_length=20
)
dataset = D4RLDatasets(dataset_config)

model_config = ModelConfig(
    n_embd=128,
    n_head=1,
    state_dim = 11,
    state_mean=dataset.obs_mean,
    state_std=dataset.obs_std,
    act_dim = 3,
    n_layer = 3,
    dropout = 0.2,
    vocab_size= 0,
    context_size = 20,
    gpt_wpe_wte_off = True,
)
dt = DecisionTransformer(model_config)
dt = dt.to(device) # type: ignore
dt = torch.compile(dt)

train_config = TrainerConfig(
    max_train_steps = 100000,
    eva_interval = 500,
    decay_lr = True,
    warmup_steps = 20000,
    learning_rate = 1e-4,
    lr_decay_steps = 90000,
    min_lr = 1e-5,
    grad_clip_max_norm = 0.25,
    weight_decay=1e-2,
    beta1=0.9,
    beta2=0.95,
    batch_size = 64,
    discrete_action=False,
    game = "HalfCheetah",
    num_eval_episode = 50,
)

run_name = f"DT_HalfCheetah_medium_expert_exp_{datetime.now().strftime('%y%m%d_%H%M')}"
wandb.init(
    project="Decision Transformer",
    name= run_name,
    config = {
        "model":model_config.__dict__,
        "train":train_config.__dict__,    
    }
)

trainer = Trainer(dt,dataset,train_config)
trainer.train()
