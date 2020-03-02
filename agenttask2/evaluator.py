import collections
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
from gym_grid_driving.envs.grid_driving import LaneSpec
from tqdm import tqdm

from models import Base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

t_max = 600

RANDOM_SEED = 5446

torch.random.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

Transition = collections.namedtuple('Transition', ('state', 'reward', 'mcts_expected', 'next_state', 'done'))


class BaseAgent(Base):
    def act(self, state, env):
        # liststate = []
        # relevantstate = []
        # for set in state[0]:
        #     liststate.append(set)
        # # self.env.render()
        # for cars in liststate:
        #     if cars.lane == state.agent.lane - 1:
        #         relevantstate.append(cars)
        # positionx = []
        # if len(relevantstate) > 0:
        #     relevantspeed = relevantstate[0].speed_range[0] * -1
        #
        # for relevantcar in relevantstate:
        #     positionx.append(relevantcar.position.x)
        #     # positionx[1].append(relevantcar.speed_range[0])
        # agentpos = state.agent.position.x
        #
        # relevantpos = []
        #
        # for pos in positionx:
        #     if ((pos - agentpos >= 0) & (pos - agentpos <= relevantspeed)) | (
        #             ((49 + pos) - agentpos >= 0) & ((49 + pos) - agentpos <= relevantspeed)):
        #         relevantpos.append(pos)
        #     # if pos - agentpos < 0 & agentpos - pos <= relevantspeed:
        #
        # backtrack = []
        # relevantsamelane = []
        #
        # for cars in liststate:
        #     if cars.lane == state.agent.lane:
        #         relevantsamelane.append(cars)
        # samelaneposx = []
        #
        # for relevantcarsamelane in relevantsamelane:
        #     samelaneposx.append(relevantcarsamelane.position.x)
        #     # positionx[1].append(relevantcar.speed_range[0])
        # # agentpos = state.agent.position.x
        #
        # relevantsamelanepos = []
        # relevantsamelanespeed = relevantsamelane[0].speed_range[0] * -1
        #
        # onestephazard = 0
        # frontline = []
        # for pos2 in samelaneposx:
        #     if ((pos2 - agentpos >= 0) & (pos2 - agentpos <= relevantsamelanespeed)) | (
        #             ((49 + pos2) - agentpos >= 0) & ((49 + pos2) - agentpos <= relevantsamelanespeed)):
        #         relevantsamelanepos.append(pos2)
        #     if ((pos2 - agentpos) < 0) & ((agentpos - pos2) <= 2):
        #         onestephazard = 1
        #
        # actionnum = 4  # forward-1
        # if (state.agent.position.y == 0) & (onestephazard == 0):
        #     actionnum = 3
        #     # print("cond1")
        # elif (len(relevantpos) == 0) & (state.agent.position.y != 0):
        #     actionnum = 0  # up
        #     # print("cond2")
        # elif (len(relevantsamelanepos) > 0) & (onestephazard != 1) & (state.agent.position.y != 0):
        #     actionnum = 3
        #     # print("cond3")
        #     if (relevantsamelanespeed > 2):
        #         actionnum = 2
        # elif (len(relevantsamelanepos) > 0) & (onestephazard != 1):
        #     actionnum = 3
        #     # print("cond4")
        # elif (state.agent.position.y != 0) and (relevantspeed == 1):
        #     actionnum = 3
        #     # print("cond5")
        #
        # if state.agent.position.x == 1:
        #     actionnum = 4
        #
        # return env.actions[actionnum]

        if not isinstance(env.world.tensor_state, torch.FloatTensor):
            state_tensor = torch.from_numpy(env.world.tensor_state).float().unsqueeze(0).to(device)
        return self.forward(state_tensor).detach().max(1)[1][0].item()


def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in tqdm(range(max_episodes), desc="Episode"):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state, env)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_path = os.path.join('./model-cars-0-ts-1573981466.3011007.pt')
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path, map_location=device)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


class DQN(BaseAgent):
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


def get_env():
    '''
    Get the sample test cases for training and testing.
    '''
    config = {'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
              }
    return gym.make('GridDriving-v0', **config)


if __name__ == '__main__':
    env = get_env()
    model = get_model()
    test(model, env, 600)
