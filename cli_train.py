from datetime import datetime
import random
import wandb
from model import DecisionTransformer,ModelConfig
from training_tools import Trainer,TrainerConfig
from dataset import D4RLDatasets,DataSetConfig
import torch
import numpy as np
import torch
from absl import app
from absl import flags

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
    
FLAGS = flags.FLAGS
flags.DEFINE_string("game","HalfCheetah","The environment the model will be train in")
flags.DEFINE_string("quality","medium_expert","Quality of the dataset")
flags.DEFINE_integer("seed",11,"Seed for the app")
flags.DEFINE_string("device","auto","Select the device for training")

def main(argv):
    gpu_info()
    setup_seed(FLAGS.seed)
    if FLAGS.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = FLAGS.device
    print(f"Using device:{device} for training")
    torch.set_float32_matmul_precision("high")

    dataset_dir = dict(
        HalfCheetah = dict(
            medium = "./data/halfcheetah-medium-v2.pkl",
            medium_expert = "./data/halfcheetah-medium-expert-v2.pkl",
            medium_replay = "./data/halfcheetah-medium-replay-v2.pkl",
        ),
        Hopper = dict(
            medium = "./data/hopper-medium-v2.pkl",
            medium_expert = "./data/hopper-medium-expert-v2.pkl",
            medium_replay = "./data/hopper-medium-replay-v2.pkl",
        )
    )
    

    dataset_config = DataSetConfig(
    path=dataset_dir[FLAGS.game][FLAGS.quality],
    sample_length=20
    )
    dataset = D4RLDatasets(dataset_config)

    if FLAGS.game == "Hopper":
        model_config =   ModelConfig(
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
    elif FLAGS.game == "HalfCheetah":
        model_config = ModelConfig(
            n_embd=128,
            n_head=1,
            state_dim = 17,
            state_mean=dataset.obs_mean,
            state_std=dataset.obs_std,
            act_dim = 6,
            n_layer = 3,
            dropout = 0.2,
            vocab_size= 0,
            context_size = 20,
            gpt_wpe_wte_off = True,
        )
    else:
        raise NotImplementedError()

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
        game = FLAGS.game,
        num_eval_episode = 50,
    )

    run_name = f"DT_{FLAGS.game}_{FLAGS.quality}_{datetime.now().strftime('%y%m%d_%H%M')}"
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

if __name__ == '__main__':
  app.run(main)