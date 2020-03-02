try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import collections
import os
import random


import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)


class DQN(Base):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )


class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_max = 600

Transition = collections.namedtuple('Transition', ('state', 'reward', 'mcts_expected', 'next_state', 'done'))



class DQNAgent(Agent):

    def __init__(self, *args, **kwargs):

        test_case_id = kwargs.get('test_case_id')


    def initialize(self, **kwargs):
        fast_downward_path = kwargs.get('fast_downward_path')
        agent_speed_range = kwargs.get('agent_speed_range')
        gamma = kwargs.get('gamma')
        self.model = get_model()

    def step(self, state, *args, **kwargs):

        if not isinstance(state, torch.FloatTensor):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return self.model.forward(state_tensor).detach().max(1)[1][0].item()


    def update(self, *args, **kwargs):
        state = kwargs.get('state')
        action = kwargs.get('action')
        reward = kwargs.get('reward')
        next_state = kwargs.get('next_state')
        done = kwargs.get('done')
        info = kwargs.get('info')


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    script_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_path, 'model8-5-2.pt')
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path, map_location=device)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def create_agent(test_case_id, *args, **kwargs):

    return DQNAgent(test_case_id=test_case_id)

