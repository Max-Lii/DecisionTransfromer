from dataclasses import dataclass
import pickle
import os
import numpy as np
import torch

@dataclass
class DataSetConfig:
    path:str
    sample_length:int

class D4RLDatasets():
    def __init__(self,config:DataSetConfig):
        if not os.path.exists(config.path):
            raise FileNotFoundError(f"Data set not exist in path:{config.path}")
        with open(config.path, 'rb') as f:
            dataset = pickle.load(f)
        self.episodes = dataset["episodes"] 
        self.n_episodes = len(self.episodes)
        self.sample_length = config.sample_length
        # sample probabilities of each episodes
        # unbalanced weights according to the length of episodes
        # so we sample according to time steps
        self.sample_probs = np.array(dataset["episode_steps"]) / float(dataset["total_steps"])

        self.obs_mean = dataset["obs_mean"]
        self.obs_std = dataset["obs_std"]
        self.s_dim = self.episodes[0]["observations"].shape[-1]
        self.a_dim = self.episodes[0]["actions"].shape[-1]
    def get_batch(self,batch_size:int):
        #sample random index to get the episodes
        sampled_ids = np.random.choice(
            np.arange(self.n_episodes),
            size = batch_size,
            replace = False,
            p = self.sample_probs
        )
        r,s,a,t,pad_mask = [],[],[],[],[]
        for i in range(batch_size):
            episode = self.episodes[sampled_ids[i]]
            #sample a fix length (sample_length) from episode
            ep_length = episode["actions"].shape[0];
            ep_start = np.random.randint(0,ep_length)
            
            # original code use left padding 
            # we use right padding instead 
            # i don't think it will cause any huge difference but more easy to implement
            #TODO put it to device if needed
            pad_length = max(0,ep_start + self.sample_length - ep_length)

            r.append(np.concatenate([episode["returns_to_go"][ep_start : ep_start+self.sample_length],np.zeros(pad_length)]))
            s.append(np.concatenate([episode["observations"] [ep_start : ep_start+self.sample_length],np.zeros((pad_length,self.s_dim))]))
            a.append(np.concatenate([episode["actions"]      [ep_start : ep_start+self.sample_length],np.zeros((pad_length,self.a_dim))]))
            t.append(self.sample_length)
            pad_mask.append(np.concatenate([np.ones(self.sample_length - pad_length),np.zeros(pad_length)]))

        #convert all r,s,a to tensor (B,T,C)
        r = torch.stack([torch.from_numpy(data) for data in r],dim=0).unsqueeze(-1).float()
        s = torch.stack([torch.from_numpy(data) for data in s],dim=0).float()
        a = torch.stack([torch.from_numpy(data) for data in a],dim=0).float()
        pad_mask = torch.stack([torch.from_numpy(data) for data in pad_mask],dim=0).float()

        return r,s,a,t,pad_mask
