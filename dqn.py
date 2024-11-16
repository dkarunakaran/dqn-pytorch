import torch
import torch.nn as nn
from torch.distributions import Categorical

# Ref: https://github.com/jacobaustin123/pytorch-dqn

class DQN(nn.Module):
    def __init__(self, state_size=None, action_size=None):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        # x is state
        output = self.relu(self.linear1(x))
        output = self.relu(self.linear2(output))

        # it outpus the Q values for actions
        actions = self.linear3(output)

        return actions
    

if __name__ == '__main__':
    dqn = DQN(10,2)
    print(dqn)
    