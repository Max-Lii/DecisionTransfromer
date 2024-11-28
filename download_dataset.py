import collections
import gym
import numpy as np
import d4rl
import pickle
from halo import Halo



datasets = []

def download_datasets(env,type):
    """
    method for downloading offline dataset from d4rl and store as parquets
    """
    name = f'{env}-{type}-v2'
    env = gym.make(name)
    spinner = Halo(text=f"processing {env}-{type} ...")
    spinner.start()
    #This is a untruncated dataset with multiple episode in it
    dataset = env.get_dataset() # type: ignore
    data_size = dataset["rewards"].shape[0]
    #steps number of a single episode
    episode_step = 1
    #length of all episode
    episode_steps = []
    max_episode_length = 1000
    episodes = []
    episode = collections.defaultdict(list)

    episode_total_reward = 0

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
    for i in range(data_size):
        terminal = bool(dataset["terminals"][i])
        if use_timeouts:
            timeout = bool(dataset['timeouts'][i])
        else:
            timeout = bool(episode_step >= max_episode_length )
        #add one step to the data
        for col in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                episode[col].append(dataset[col][i])
        episode_total_reward += dataset["rewards"][i]
        #end of the episode
        if terminal or timeout or i == data_size-1 :
            episode_np = {}
            returns_to_go = []
            for r in range(episode_step):
                episode_total_reward -= episode["rewards"][r]
                returns_to_go.append(episode_total_reward)
            episode["returns_to_go"] = returns_to_go
            for k in episode:
                episode_np[k] = np.array(episode[k])
            episodes.append(episode_np)
            episode = collections.defaultdict(list)
            episode_total_reward = 0
            episode_steps.append(episode_step)
            episode_step = 1
        else:
            episode_step +=1

    #store the data set with some meta information
    episodes_with_meta = dict(
        episodes = episodes,
        episode_steps = np.array(episode_steps),
        total_steps = len(dataset["observations"]),
        obs_mean = np.mean(dataset['observations'],axis=0),
        obs_std  = np.std(dataset['observations'],axis=0) + 1e-6
    )
    with open(f'./data/{name}.pkl', 'wb') as f:
         pickle.dump(episodes_with_meta, f)
    spinner.succeed(f"{env}-{type} download success")
       
if __name__ == "__main__":
    envs = ["hopper","halfcheetah"]
    types = ["medium","medium-replay","medium-expert"]
    for e in envs:
        for t in types:
            download_datasets(e,t)
