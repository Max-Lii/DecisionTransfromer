
from dataclasses import dataclass
import math
import sys
import torch
from torch.nn import functional as F 
from torch.nn import Module
from torch import Tensor
from typing import Tuple
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import model
from halo import Halo
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
    weight_decay:float = 1e-2
    beta1:float = 0.9
    beta2:float = 0.999
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
            weight_decay=config.weight_decay,
            betas=(config.beta1,config.beta2)
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
            self.loss_fn = lambda pred_action,target_action : F.mse_loss(pred_action,target_action)#torch.mean((pred_action - target_action)**2) 
        else:
            # using cross entropy error loss for continue action
            self.loss_fn = lambda pred_action,target_action : F.cross_entropy(pred_action.view(-1,pred_action.size(-1)),target_action.view(-1))
        if self.game == "Hopper": 
            self.evaluator = Evaluator(config)
            #scaling the reward
            self.rtg_scale = 1800
            pass
        elif self.game == "HalfCheetah":
            self.evaluator = Evaluator(config)
            self.rtg_scale = 1000
        #support more games if needed
        else:
            raise NotImplementedError()

        self.spinner = Halo(text="training...",spinner="dots",stream=sys.stdout)
    def log(self,text:str):
        self.spinner.stop()
        print(text)
        self.spinner.start()
        pass


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
        loss = self.loss_fn(pred_a,a.detach())
        loss = (loss * pad_mask.unsqueeze(-1)).mean()
        optimizer.zero_grad()
        # TODO:accelerate
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
        mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step =  self.evaluator.parallelEvaluate(model)
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
        
        self.spinner.start()
        model:Module = self.model
        optimizer = self.optimizer
        training_loss = []

        #training loop
        for step in range(self.max_train_steps):
            loss = self.train_step(step)
            training_loss.append(loss.item())

            if (step+1) % self.eva_interval == 0:
                self.spinner.text="evaluating..."
                mean_train_loss = np.mean(training_loss)
                self.log(f"current steps:{step} lr:{'{:.6f}'.format(self.get_lr(step))}  training loss:{'{:.3f}'.format(mean_train_loss)}")
                mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step = self.eval(model) 
                self.log(f"current evaluate: mean_steps:{'{:.3f}'.format(mean_steps)} mean_rewards:{'{:.3f}'.format(mean_rewards)} r/step:{'{:.3f}'.format(mean_rewards/mean_steps)}  min_r:{'{:.3f}'.format(min_reward)} max_r:{'{:.3f}'.format(max_reward)} min_stp:{'{:.3f}'.format(min_step)} max_stp:{'{:.3f}'.format(max_step)}")
                training_loss = []

                wandb.log(
                    step = step,
                    data = dict(
                        train_loss = mean_train_loss,
                        lr = self.get_lr(step),
                        avg_eval_steps = mean_steps,
                        avg_eval_return = mean_rewards,
                        return_per_step = mean_rewards/mean_steps,
                        min_eval_steps = min_step,
                        max_eval_steps = max_step,
                        min_eval_return = min_reward,
                        max_eval_return = max_reward,
                    )
                )
                self.spinner.text="training..."
        self.spinner.succeed("train finish!")
        pass

class Evaluator():
    def __init__(self,config:TrainerConfig):
        self.game = config.game
        self.num_eval_episode = config.num_eval_episode
        self.seeds = np.random.randint(100000,size=self.num_eval_episode)
        if config.game == "Hopper":
            self.env = gym.make("Hopper-v3",render_mode=None)
            self.envs = AsyncVectorEnv([lambda:gym.make("Hopper-v3") for _ in range(self.num_eval_episode)])
            #Original code use both 3600 and 1800 now we just keep it simple
            self.target_return = 3600.0
            self.return_scale  = 1800.0
            self.max_ep_steps = 1000
        elif config.game == "HalfCheetah":
            self.env = gym.make("HalfCheetah-v3",render_mode=None)
            self.envs = AsyncVectorEnv([lambda:gym.make("HalfCheetah-v3") for _ in range(self.num_eval_episode)])
            self.target_return = 12000.0
            self.return_scale  = 1000.0
            self.max_ep_steps = 1000
        else:
            raise NotImplementedError
    def evaluate(self,model:model.DecisionTransformer)->Tuple[float,float,float,float,float,float,]:
        """
            evaluate the model with and
            return the mean steps and mean reward during the games
        """
        device = next(model.parameters()).device
        env:gym.Env = self.env
        act_dim = env.action_space.sample().shape[0]
        state_dim = env.observation_space.sample().shape[0]
        total_steps = []
        total_rewards =[]
        min_reward = 10000
        max_reward = -10000
        min_step   = 10000
        max_step   = 0
        #Evaluate 10 turns
        for i in range(self.num_eval_episode):
            ini_state,info = env.reset(seed=int(self.seeds[i]))
            rtg:Tensor     = torch.tensor([self.target_return/self.return_scale],device=device)
            states:Tensor  = torch.tensor(np.array([ini_state]),device=device,dtype=torch.float32)
            actions:Tensor = torch.zeros((act_dim),device=device)
            time_steps = [1]
            
            #steps and reward for the whole episode
            ep_step = 0
            ep_reward = 0 
            while True:
                # sample the action from model
                pred_act = model.get_action(
                    returns_to_go = rtg    .view(1,-1,1),
                    states        = states .view(1,-1,state_dim),
                    actions       = actions.view(1,-1,act_dim),
                    time_steps=time_steps,
                )[0]
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

    def parallelEvaluate(self,model)->Tuple[float,float,float,float,float,float,]:
        device = next(model.parameters()).device
        num_episode = self.num_eval_episode
        batch_size = 64

        # one episodes stand for one env
        envs = self.envs
        act_dim = envs.action_space.sample().shape[1]
        #state_dim = envs.observation_space.sample().shape[1]

        terminate_count = 0
        ini_state,info = envs.reset(seed=self.seeds.tolist())

        target_return = self.target_return
        return_scale = self.return_scale
        ini_rtg_value = target_return / return_scale
        current_step = 1

        actions = [[np.zeros(act_dim)] for _ in range(num_episode)] #[[] for _ in range(num_episode)]#
        states  = [[ini_state[i]] for i in range(num_episode)]
        rtgs    = [[ini_rtg_value] for _ in range(num_episode)]

        episode_terminate = [False for _ in range(num_episode)]
        episode_returns = [0 for _ in range(num_episode)]
        episode_steps = [0 for _ in range(num_episode)]

        # Loop for evaluation
        while True:
            ep_i = 0
            # Loop for get a single step action for all episodes
            while True:
                # Get a prediction for all ongoing envs in a batch
                batch_rtg = []
                batch_state = []
                batch_action = []
                batch_ep_index =[]
                gather_size = 0
                # Loop for gathering ongoing episode's r,s,a,t into a batch for GPU to predict next action 
                while True:
                    # If the i th env is terminate we give it a radom action for alignment
                    # and skip gathering it into the batch
                    if episode_terminate[ep_i]:
                        # Just give a radom action it doesn't matter since it has finished
                        actions[ep_i].append(np.zeros(act_dim))
                        ep_i += 1
                        if ep_i >= num_episode:
                            break
                        continue

                    # If i th env is still on going, gathering it into the batch
                    batch_rtg.append(rtgs[ep_i])
                    batch_state.append(states[ep_i])
                    batch_action.append(actions[ep_i])
                    batch_ep_index.append(ep_i)
                    gather_size +=1
                    ep_i+=1

                    # Gather until we meet the batch size or the maximum size of episodes
                    if ep_i >= num_episode or gather_size >= batch_size:
                        break
                # Send the batch to model to get the predict actions
                # print(f"{batch_action[0]=}")

                batch_time   = [current_step for _ in range(len(batch_rtg))] 
                batch_rtg    = torch.from_numpy(np.array(batch_rtg)).unsqueeze(-1).float().to(device)
                batch_state  = torch.from_numpy(np.array(batch_state)).float().to(device)
                batch_action = torch.from_numpy(np.array(batch_action)).float().to(device)

                pred_acts = model.get_action(
                    returns_to_go=batch_rtg,
                    states=batch_state,
                    actions=batch_action,
                    time_steps=batch_time,
                ).detach().cpu().numpy()

                #Loop for store the predict action to the corresponding actions
                for batch_index,env_index in enumerate(batch_ep_index):
                    actions[env_index].insert(-1,pred_acts[batch_index])
                    # actions[env_index].append(pred_acts[batch_index])
                
                # finish getting the action for all episodes
                if ep_i >= num_episode:
                    break
            
            # Make a step for all episodes (envs) 
            step_actions = [ action[-2] for action in actions ]
            # print(step_actions)
            observations, rewards, terminates, _, _ = envs.step(step_actions)
            # print(rewards)

            
            for i in range(num_episode):
                states[i].append(observations[i])

                if episode_terminate[i]:
                    continue
                
                # If episode got terminate set the terminate flag to true
                if terminates[i]:
                    terminate_count += 1
                    episode_terminate[i] = True

                rtgs[i].append(rtgs[i][-1] - (rewards[i] / return_scale))
                episode_returns[i] += rewards[i]
                episode_steps[i] += 1
              
            current_step += 1
            # if all episodes are terminated finish the evaluation
            if terminate_count >= num_episode or current_step > self.max_ep_steps:
                break
            
        mean_steps = float(np.mean(episode_steps))
        mean_rewards = float(np.mean(episode_returns))
        min_reward = float(np.min(episode_returns))
        max_reward = float(np.max(episode_returns))
        min_step = float(np.min(episode_steps))
        max_step = float(np.max(episode_steps))

        return mean_steps,mean_rewards,min_reward,max_reward,min_step,max_step