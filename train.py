import torch
import torch.optim as optim
import gymnasium as gym 
from dqn import DQN
import yaml
from utility import logger_helper, ReplayBuffer
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ref: https://github.com/jacobaustin123/pytorch-dqn/
# Ref" https://medium.com/@vignesh.g1609/deep-q-learning-dqn-using-pytorch-a31f02a910ac
# Ref for paying the cartpole with trained model: https://github.com/Rabrg/dqn/blob/main/demo.ipynb


# ***********************************Important thing to remeber************************************
'''To ensure steady learning and treat Reinforcement Learning (RL) as supervised learning with Independent and 
Identically Distributed (IID) assumptions and a stable target, we employ a replay buffer and a second network as a target network.

The target network provides Q-values for computing the actual return in the loss function.

The prediction network, which selects actions using an epsilon-greedy approach, is updated every time.

Loss function = r+gamma*max(Q(s_t+1, a_t+1)) - Q(s_t,a_t) value from prediction network, where Q value in the max function is computed by
traget network and other Q value is predicted by prediction network  

To maintain a stable target, the target network is not updated frequently.

We use the same network code for both networks, and the prediction network's weights are loaded into the target network at specified intervals'''
# *************************************************************************************************

class Train:
    def __init__(self):
        self.random_seed = 543
        self.env = gym.make('CartPole-v1')
        observation, info = self.env.reset()
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.logger = logger_helper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.prediction = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.replay_buffer = ReplayBuffer()
        

    def run(self):
        self.logger.info("Training started")

        # loop through the episode
        # loop through timespteps
        # sample the action
        # collect the expereince in replay_buffer
        # once replay_buffer reaches the threshold fixed, train the prediction network. We can use l1_smooth_loss with actual return(r+max(Q_target)) and predicted return(Q_predction)
        # Once a threshold reached with iteration, load the prediction NN weights to target NN
        # Store the plotting results
        # Visualize the results to see how the training going
        pass


if __name__ == '__main__':
    train = Train()
    train.run()