
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F 
from torch.nn import Module
from torch import Tensor
from typing import Tuple
import numpy as np
import gymnasium as gym
import model

@dataclass
class TrainerConfig:
    max_train_steps:int
    eva_interval:int
    log_interval:int
    decay_lr:bool
    warmup_steps:int
    learning_rate:int
    lr_decay_steps:int
    min_lr:int
    batch_size:int
    discrete_action:bool
    game:str


class Trainer():
    """
        This is the trainer for the Decision Transformer,
        For now only supporting Mujoco(hopper) game
    """
    def __init__(self,model:Module,dataloader,config:TrainerConfig):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.batch_size = config.batch_size
        self.max_train_steps = config.max_train_steps
        self.eva_interval = config.eva_interval
        self.log_interval = config.log_interval
        
        #learning rate config
        self.decay_lr = config.decay_lr
        self.warmup_steps = config.warmup_steps
        self.learning_rate = config.learning_rate
        self.lr_decay_steps = config.lr_decay_steps
        self.min_lr = config.min_lr

        #environment config
        self.game = config.game
        
        if not config.discrete_action:
            # using mean square error loss for continue action
            self.loss_fn = lambda pred_action,target_action : torch.mean((pred_action - target_action)**2) 
        else:
            # using cross entropy error loss for continue action
            self.loss_fn = lambda pred_action,target_action : F.cross_entropy(pred_action.view(-1,pred_action.size(-1)),target_action.view(-1))
        if self.game == "Hopper": 
            #self.evaluator = Evaluator(config)
            pass
        #support more games if needed
        else:
            raise NotImplementedError()
        
        

    def train_step(self,step):
        optimizer = self.optimizer
        model = self.model
        lr = self.get_lr(step) if self.decay_lr else self.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # load data from dataloader
        r,s,a,t,pad_mask = self.dataloader.get_batch(self.batch_size)
        # According to original paper only use pred_a for optimization
        pred_r,pred_s,pred_a = model(r,s,a,t,pad_mask)

        # Because pred_a is predict from current s as the next action (1 token behind) which can't see the real action
        loss = self.loss_fn(pred_a,a)
        optimizer.zero_grad()
        # TODO:accelerator
        loss.backward()
        optimizer.step()
        return loss

    def eval(self,model)->Tuple[float,float]:
        #if using accelerator the model might need to be preprocessing, not sure for now 
        mean_steps,mean_rewards =  self.evaluator.evaluate(model)
        return mean_steps,mean_rewards

    def get_lr(self,step):
        """
            Cosine decay learning rate with warmup
        """
         # 1) linear warmup for warmup_iters steps
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if step > self.lr_decay_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (self.lr_decay_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        cos_val = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cos_val ranges from 0 to 1
        return self.min_lr + cos_val * (self.learning_rate - self.min_lr)
    
    def train(self):
        model = self.model
        optimizer = self.optimizer
        training_loss = []

        #training loop
        for step in range(self.max_train_steps):
            loss = self.train_step(step)
            training_loss.append(loss.item())
            #TODO:logging

            if (step+1) % self.eva_interval == 0:
               #TODO:logging
               mean_train_loss = np.mean(training_loss)
               print(f"current steps:{step}  training loss:{'{:.3f}'.format(mean_train_loss)}")
               training_loss = []
               print("start evaluation...")
            #    mean_steps,mean_rewards = self.eval(model) 
            #    print(f"evaluate result: mean steps:{'{:.3f}'.format(mean_steps)} mean rewards{'{:.3f}'.format(mean_rewards)}")
               
        pass

class Evaluator():
    def __init__(self,config:TrainerConfig):
        self.game = config.game
        if config.game == "Hopper":
            self.env = gym.make("Hopper-v2")
            #Original code use both 3600 and 1800 now we just keep it simple
            self.target_return = 3600
            self.return_scale  = 1000
        else:
            raise NotImplementedError
    def evaluate(self,model:model.DecisionTransformer)->Tuple[float,float]:
        """
            evaluate the model with 10 episodes and
            return the mean steps and mean reward during the games
        """

        env:gym.Env = self.env
        total_steps = []
        total_rewards =[]
        #Evaluate 10 turns
        for _ in range(10):
            ini_state,info = env.reset()
            rtg:Tensor = torch.Tensor([self.target_return])
            states:Tensor = torch.Tensor([ini_state])
            actions:Tensor = torch.Tensor([])
            time_steps = [0]
            #steps and reward for the whole episode
            ep_step = 0
            ep_reward = 0 
            while True:
                # sample the action from model
                pred_act = model.get_action(
                    returns_to_go=rtg,
                    states=states,
                    actions=actions,
                    time_steps=time_steps
                )
                new_state,reward,terminated =env.step(pred_act.detach().cpu().numpy())
                # rtg[t] = rtg[t-1] - rewards
                new_rtg = rtg[-1].item() - reward/self.return_scale
                rtg   = torch.cat((rtg,new_rtg),dim=0)
                states = torch.cat((states,torch.tensor(new_state)),dim=0)
                actions= torch.cat((actions,torch.tensor(pred_act)),dim=0)
                time_steps.append(time_steps[-1]+1)

                ep_step +=1
                ep_reward += reward
                
                if terminated:
                    break
            total_steps.append(ep_step)
            total_rewards.append(ep_reward)
        
        mean_steps = np.mean(total_steps)
        mean_rewards = np.mean(total_rewards)

        return mean_steps,mean_rewards

    def cleanup(self):
        self.env.close()