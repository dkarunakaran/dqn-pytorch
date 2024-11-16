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
import torch.nn as nn
import random
import math

# Ref: https://github.com/jacobaustin123/pytorch-dqn/
# Ref: https://medium.com/@vignesh.g1609/deep-q-learning-dqn-using-pytorch-a31f02a910ac
# Ref: https://github.com/dxyang/DQN_pytorch/blob/master/learn.py
# Ref: https://www.akshaymakes.com/blogs/deep_q_learning
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
        self.prediction = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target.load_state_dict(self.prediction.state_dict())
        self.replay_buffer = ReplayBuffer(capacity=self.cfg['replay_buffer_size'])
        self.epsilon_start = self.cfg['train']['epsilon_start']
        self.epsilon_end = self.cfg['train']['epsilon_end']
        self.epsilon_decay = self.cfg['train']['epsilon_decay']
        self.steps_done = 1
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay) 
        if sample > eps_threshold:
            with torch.no_grad():
               return torch.argmax(self.prediction(state))
        else:
            return torch.tensor(self.env.action_space.sample())

    def run(self):
        self.logger.info("Training started")

        torch.manual_seed(self.cfg['train']['random_seed'])
        optim_fn = optim.Adam(self.prediction.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])

        # loop through the episode
        for episode in range(self.cfg['train']['n_epidode']):
            self.logger.info(f"--------Episode: {episode} started--------step done:{self.steps_done}")
            state = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state[0])
            done = False
            optim_fn.zero_grad()
            count = 1
            # loop through timespteps
            while not done:
                # Sample the action
                action = self.select_action(state)
                next_state, reward, done, _, _= self.env.step(action.item())
                next_state = torch.FloatTensor(next_state)
                # Collect the expereince in replay_buffer
                self.replay_buffer.push(state, action, next_state, reward)

                # Assign next state as current state
                state = next_state

                # Update prediction network
                if self.steps_done%self.cfg['train']['update_freq'] == 0:
                    # Loss function = r+gamma*max(Q(s_t_1, a)) - Q(s_t,a)
                    # We need to Compute Q(s_t, a) for the loss function, the prediction model computes Q(s_t), 
                    # then we select the the Q values for the action taken
                    replay_data = self.replay_buffer.sample(self.cfg['train']['batch_size'])
                    #print(replay_data)
                    for state, action, next_state, reward in replay_data:
                        # Get the Q values for current observations (Q(s_t,a)) from prediction network
                        q_values_st = self.prediction(state)
                        # We got the Q values for all the action for the given state 
                        # and now getting the q_value for slected action
                        q_s_a = torch.gather(q_values_st, 0, action) 
                        #max(Q(s_t_1, a)) - from target network
                        q_values_st_1 = torch.max(self.target(next_state))
                        # actual_return - r+gamma*max(Q(s_t_1, a))
                        actural_return = reward+(self.cfg['train']['gamma']+q_values_st_1)
                        #actural_return.requires_grad = True
                        # Huber loss, which is less sensitive to outliers in data than squared-error loss. In value based RL ssetup, huber loss is preferred.
                        # Smooth L1 loss is closely related to HuberLoss
                        loss =  F.smooth_l1_loss(q_s_a, actural_return)
                        loss.backward()
                        optim_fn.step()

                    self.logger.info(f"Updated the prediction network at {self.steps_done}")
                
                # Update target network
                if self.steps_done%self.cfg['train']['target_update_freq'] == 0:
                    self.target.load_state_dict(self.prediction.state_dict())
                    self.logger.info(f"Updated the target network at {self.steps_done}")

                # Enviornment return done == true if the current episode is terminated
                if done:
                    self.logger.info('Iteration: {}, Score: {}'.format(episode, count))
                    break
                count += 1
                self.steps_done += 1   

if __name__ == '__main__':
    train = Train()
    train.run()