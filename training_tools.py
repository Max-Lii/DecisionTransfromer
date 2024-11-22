
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
import wandb

@dataclass
class TrainerConfig:
    max_train_steps:int
    eva_interval:int
    decay_lr:bool
    warmup_steps:int
    learning_rate:float
    lr_decay_steps:int
    min_lr:float
    batch_size:int
    discrete_action:bool
    game:str
    num_eval_episode:int = 50
    grad_clip_max_norm:float|None = None



class Trainer():
    """
        This is the trainer for the Decision Transformer,
        For now only supporting Mujoco(hopper) game
    """
    def __init__(self,model,dataloader,config:TrainerConfig):
        self.dataloader = dataloader
        self.model = model
        self.device = next(self.model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.grad_clip_max_norm = config.grad_clip_max_norm
        self.batch_size = config.batch_size
        self.max_train_steps = config.max_train_steps
        self.eva_interval = config.eva_interval
        
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
            self.evaluator = Evaluator(config)
            #scaling the reward
            self.rtg_scale = 1800
            pass
        #support more games if needed
        else:
            raise NotImplementedError()
        
        

    def train_step(self,step):
        device = self.device
        optimizer = self.optimizer
        model:Module = self.model
        lr = self.get_lr(step) if self.decay_lr else self.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # load data from dataloader to device
        r,s,a,t,pad_mask = self.dataloader.get_batch(self.batch_size)
        r = r / self.rtg_scale
        r,s,a,pad_mask  = r.to(device),s.to(device),a.to(device),pad_mask.to(device) 
        # According to original paper only use pred_a for optimization
        pred_r,pred_s,pred_a = model(r,s,a,t,pad_mask)

        # Because pred_a is predict from current s as the next action (1 token behind) which can't see the real action
        loss = self.loss_fn(pred_a,a)
        optimizer.zero_grad()
        # TODO:accelerator
        loss.backward()
        # clip the grad
        if self.grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),self.grad_clip_max_norm)
        optimizer.step()
        return loss
    @torch.no_grad()
    def eval(self,model)->Tuple[float,float,float,float,float,float,]:
        #if using accelerator the model might need to be preprocessing, not sure for now 
        model.eval()
        mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step =  self.evaluator.evaluate(model)
        model.train()
        return mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step

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
        model:Module = self.model
        optimizer = self.optimizer
        training_loss = []

        #training loop
        for step in range(self.max_train_steps):
            loss = self.train_step(step)
            training_loss.append(loss.item())

            if (step+1) % self.eva_interval == 0:
                mean_train_loss = np.mean(training_loss)
                print(f"current steps:{step} lr:{'{:.6f}'.format(self.get_lr(step))}  training loss:{'{:.3f}'.format(mean_train_loss)}")
                training_loss = []
                mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step = self.eval(model) 
                print(f"current evaluate: mean_steps:{'{:.3f}'.format(mean_steps)} mean_rewards:{'{:.3f}'.format(mean_rewards)} r/step:{'{:.3f}'.format(mean_rewards/mean_steps)}  min_r:{'{:.3f}'.format(min_reward)} max_r:{'{:.3f}'.format(max_reward)} min_stp:{'{:.3f}'.format(min_step)} max_stp:{'{:.3f}'.format(max_step)}")

                wandb.log(
                    step = step,
                    data = dict(
                        train_loss = mean_train_loss,
                        lr = self.get_lr(step),
                        eval_mean_steps = mean_steps,
                        eval_mean_return = mean_rewards,
                        eval_steps_range = [min_step,max_step],
                        eval_return_range = [min_reward,max_reward],
                    )
                )

        pass

class Evaluator():
    def __init__(self,config:TrainerConfig):
        self.game = config.game
        self.num_eval_episode = config.num_eval_episode
        self.seeds = np.random.randint(1, size=self.num_eval_episode)
        if config.game == "Hopper":
            self.env = gym.make("Hopper-v3",render_mode=None)
            #Original code use both 3600 and 1800 now we just keep it simple
            self.target_return = 3600.0
            self.return_scale  = 1800.0
            self.max_ep_steps = 1000
        else:
            raise NotImplementedError
    def evaluate(self,model:model.DecisionTransformer)->Tuple[float,float,float,float,float,float,]:
        """
            evaluate the model with 10 episodes and
            return the mean steps and mean reward during the games
        """
        device = next(model.parameters()).device
        env:gym.Env = self.env
        act_dim = env.action_space.sample().shape[0]
        total_steps = []
        total_rewards =[]
        min_reward = 10000
        max_reward = 0
        min_step   = 10000
        max_step   = 0
        #Evaluate 10 turns
        for i in range(self.num_eval_episode):
            ini_state,info = env.reset(seed=int(self.seeds[i]))
            rtg:Tensor = torch.tensor([self.target_return/self.return_scale],device=device)
            states:Tensor = torch.tensor(np.array([ini_state]),device=device,dtype=torch.float32)
            actions:Tensor = torch.zeros((act_dim),device=device)
            time_steps = [1]
            
            #steps and reward for the whole episode
            ep_step = 0
            ep_reward = 0 
            while True:
                # sample the action from model
                pred_act = model.get_action(
                    returns_to_go=rtg,
                    states=states,
                    actions=actions,
                    time_steps=time_steps,
                )
                observation, reward, terminated, truncated, info =env.step(pred_act.detach().cpu().numpy())
                # rtg[t] = rtg[t-1] - rewards
                new_rtg = rtg[-1].item() - float(reward)/self.return_scale
                rtg   =  torch.cat([rtg,torch.tensor([new_rtg],device=device)],dim=0)
                states = torch.cat((states,torch.tensor(np.array([observation]),dtype=torch.float32,device=device)),dim=0)
                actions= torch.cat((actions,pred_act),dim=0)
                time_steps[0] += 1

                ep_step += 1
                ep_reward += float(reward)
                
                if terminated or ep_step >= self.max_ep_steps:
                    min_reward = min(min_reward,ep_reward)
                    max_reward = max(max_reward,ep_reward)
                    min_step   = min(min_step,ep_step)
                    max_step   = max(max_step,ep_step)
                    break
            total_steps.append(ep_step)
            total_rewards.append(ep_reward)
        
        mean_steps = float(np.mean(total_steps))
        mean_rewards = float(np.mean(total_rewards))

        return mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step

    def cleanup(self):
        self.env.close()