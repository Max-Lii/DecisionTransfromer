import os
import pickle
import numpy as np
from halo import Halo

def clean_dataset(target_path:str,store_path:str):
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Data set not exist in path:{target_path}")
    with open(target_path, 'rb') as f:
        dataset = pickle.load(f)
    spinner = Halo(text=f"processing ...")
    spinner.start()
    remove_index = []
    episodes:list = dataset["episodes"]
    episode_steps:np.ndarray = dataset["episode_steps"]

    min_rewards = -100

    #Find the data which return lower than min rewards 
    for i in range(len(episodes)):
        if episodes[i]["returns_to_go"][0] < min_rewards:
            remove_index.append(i)

    # #filter the dataset and recalculate some statistics
    episode_steps = np.delete(episode_steps,remove_index)
    episodes = [e for i, e in enumerate(episodes) if i not in remove_index]
    total_steps = np.sum(episode_steps)

    new_episodes_with_meta = dict(
        episodes = episodes,
        episode_steps = np.array(episode_steps),
        total_steps = total_steps,
        obs_mean = dataset["obs_mean"],
        obs_std  = dataset["obs_std"],
    )
    with open(store_path, 'wb') as f:
        pickle.dump(new_episodes_with_meta, f)
    spinner.succeed(f"dataset cleaning success")
    # print(f"{np.sum(episode_steps)=}")
    # print(f"{len(episode_steps)=}")
    # print(f"{len(dataset['episodes'])=}")
    # print(f"{len(episodes)=}")
    # print(f"{total_steps=}")
    

if __name__ == "__main__":
    clean_dataset("./data/halfcheetah-medium-replay-v2.pkl","./data/halfcheetah-medium-replay-clean-v2.pkl")
    
