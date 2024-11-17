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
        self.steps_done = 1
    
    def select_action(self, state, epsilon):
        if torch.rand(1) < epsilon:
            self.logger.debug("Action selected from the prediction network")
            with torch.no_grad():
               return torch.argmax(self.prediction(state))
        else:
            self.logger.debug(f"Action selected randomly, epsilon: {epsilon}")
            return torch.tensor(self.env.action_space.sample())

    def run(self):
        self.logger.info("Training started")

        torch.manual_seed(self.cfg['train']['random_seed'])
        optim_fn = optim.Adam(self.prediction.parameters(), lr=self.cfg['train']['lr'])

        epsilon = self.cfg['train']['epsilon']
        stats = self.cfg['train']['stats']
        # loop through the episode
        for episode in range(self.cfg['train']['n_epidode']):
            self.logger.debug(f"--------Episode: {episode} started--------step done:{self.steps_done}")
            state = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state[0])
            done = False
            optim_fn.zero_grad()
            count = 1
            truncated, terminated = False, False # initiate the terminated and truncated flags
            self.prediction.train()
            self.target.train()
            # loop through timespteps
            while not truncated and not terminated:
                # Sample the action
                action = self.select_action(state, epsilon)
                next_state, reward, truncated, terminated, _= self.env.step(action.item())

                next_state = torch.FloatTensor(next_state)
                # Collect the expereince in replay_buffer
                self.replay_buffer.push([state, action, reward, truncated,terminated, next_state])

                # Assign next state as current state
                state = next_state

                # Update prediction network
                if self.steps_done%self.cfg['train']['update_freq'] == 0:
                    # Loss function = r+gamma*max(Q(s_t_1, a)) - Q(s_t,a)
                    # We need to Compute Q(s_t, a) for the loss function, the prediction model computes Q(s_t), 
                    # then we select the the Q values for the action taken
                    replay_data = self.replay_buffer.sample(self.cfg['train']['batch_size'])
                    #print(replay_data)
                    for state_b, action_b, reward_b, truncated_b, terminated_b, next_state_b in replay_data:
                        # Get the Q values for current observations (Q(s_t,a)) from prediction network
                        q_values_st = self.prediction(state_b)
                        # We got the Q values for all the action for the given state 
                        # and now getting the q_value for slected action
                        q_s_a = torch.gather(q_values_st, 0, action_b) 
                        #max(Q(s_t_1, a)) - from target network
                        q_values_st_1 = torch.max(self.target(next_state_b), dim=-1, keepdim=True)[0]
                        # actual_return - r+gamma*max(Q(s_t_1, a))
                        actural_return = reward_b+(~(truncated_b + terminated_b) * self.cfg['train']['gamma']+q_values_st_1)
                        #actural_return.requires_grad = True
                        # Huber loss, which is less sensitive to outliers in data than squared-error loss. In value based RL ssetup, huber loss is preferred.
                        # Smooth L1 loss is closely related to HuberLoss
                        self.prediction.zero_grad()
                        loss =  F.smooth_l1_loss(q_s_a, actural_return)
                        stats['loss'].append(loss.item())
                        loss.backward()
                        optim_fn.step()

                    self.logger.debug(f"Updated the prediction network at {episode} episode")

                # Enviornment return done == true if the current episode is terminated
                if truncated or terminated:
                    self.logger.info('Episode: {}, Score: {}'.format(episode, count))
                count += 1
                self.steps_done += 1   

            epsilon = max(0, epsilon - 1/10000)

            # Update target network
            if episode%self.cfg['train']['target_update_freq'] == 0:
                self.target.load_state_dict(self.prediction.state_dict())
                self.logger.info(f"Updated the target network at {episode} episode")

            # Storing average losses for plotting
            if episode%self.cfg['train']['store_stats'] == 0:
                stats['avg_loss'].append(np.mean(stats['loss']))
                stats['loss'] = []

        torch.save(self.prediction, 'model/prediction.pkl')
        torch.save(self.target, 'model/target.pkl')

        plt.figure(figsize=(10,6))
        plt.xlabel("X-axis")  # add X-axis label
        plt.ylabel("Y-axis")  # add Y-axis label
        plt.title("Loss")  # add title
        plt.plot(stats['avg_loss'])
        plt.savefig('loss.png')
        plt.close()

if __name__ == '__main__':
    train = Train()
    train.run()