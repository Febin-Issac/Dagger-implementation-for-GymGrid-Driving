import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import SmoothL1Loss
from models import Base
from copy import deepcopy
from tqdm import tqdm
from gym_grid_driving.envs.grid_driving import DenseReward, SparseReward
import time

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma = 0.98
buffer_limit = 5000
batch_size = 32
max_episodes = 2000
t_max = 600
min_buffer = 2000
target_update = 20  # episode(s)
train_steps = 10
max_epsilon = 1.03
min_epsilon = 0.01
epsilon_decay = 500
print_interval = 20

RANDOM_SEED = 5446

torch.random.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

Transition = collections.namedtuple('Transition', ('state', 'reward', 'mcts_expected', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = []
        self.buffer_limit = buffer_limit
        self.next_idx = 0

    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        if len(self.buffer) < self.buffer_limit:
            self.buffer.append(None)
        self.buffer[self.next_idx] = transition
        self.next_idx = (self.next_idx + 1) % self.buffer_limit

    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `mcts_expected`     (`torch.tensor` [batch_size, 1])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        data = random.sample(self.buffer, batch_size)
        vals = Transition(*zip(*data))
        states = torch.tensor(vals[0], dtype=torch.float, device=device)
        rewards = torch.tensor(vals[1], dtype=torch.float, device=device)
        expected_action = torch.tensor(vals[2], dtype=torch.long, device=device)
        next_states = torch.tensor(vals[3], dtype=torch.float, device=device)
        dones = torch.tensor(vals[4], dtype=torch.float, device=device)

        return states, rewards, expected_action, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class BaseAgent(Base):
    def act(self, state, epsilon=0.0, episode=0):
        if not isinstance(state, torch.FloatTensor):
            # state_tensor = np.copy(env.world.tensor_state)
            state_tensor = torch.from_numpy(env.world.tensor_state).float().unsqueeze(0).to(device)

        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''

        # act_idx = random.randrange(0, 5)
        # if episode < 200:  # initially bias towards up
        #     act_p = random.uniform(0.0, 1.0)
        #     if act_p > 0.7:  # 0 -> 0.7 => random action ; 0.7 -> 1 => up with certainity
        #         act_idx = 0  # up

        liststate = []
        relevantstate = []
        for set in state[0]:
            liststate.append(set)
        # self.env.render()
        for cars in liststate:
            if cars.lane == state.agent.lane - 1:
                relevantstate.append(cars)
        positionx = []
        if len(relevantstate) > 0:
            relevantspeed = relevantstate[0].speed_range[0] * -1

        for relevantcar in relevantstate:
            positionx.append(relevantcar.position.x)
            # positionx[1].append(relevantcar.speed_range[0])
        agentpos = state.agent.position.x

        relevantpos = []

        for pos in positionx:
            if ((pos - agentpos >= 0) & (pos - agentpos < relevantspeed)) | (
                    ((49 + pos) - agentpos >= 0) & ((49 + pos) - agentpos < relevantspeed)):
                relevantpos.append(pos)
            # if pos - agentpos < 0 & agentpos - pos <= relevantspeed:

        backtrack = []
        relevantsamelane = []

        for cars in liststate:
            if cars.lane == state.agent.lane:
                relevantsamelane.append(cars)
        samelaneposx = []

        for relevantcarsamelane in relevantsamelane:
            samelaneposx.append(relevantcarsamelane.position.x)
            # positionx[1].append(relevantcar.speed_range[0])
        # agentpos = state.agent.position.x

        relevantsamelanepos = []
        relevantsamelanespeed = relevantsamelane[0].speed_range[0] * -1

        onestephazard = 0
        frontline = []
        for pos2 in samelaneposx:
            if ((pos2 - agentpos >= 0) & (pos2 - agentpos <= relevantsamelanespeed)) | (
                    ((49 + pos2) - agentpos >= 0) & ((49 + pos2) - agentpos <= relevantsamelanespeed)):
                relevantsamelanepos.append(pos2)
            if ((pos2 - agentpos) < 0) & ((agentpos - pos2) < 2):
                onestephazard = 1

        actionnum = 4  # forward-1
        if (state.agent.position.y == 0) & (onestephazard == 0):
            actionnum = 3
            # print("cond1")
        elif (len(relevantpos) == 0) & (state.agent.position.y != 0):
            actionnum = 0  # up
            # print("cond2")
        elif (len(relevantsamelanepos) > 0) & (onestephazard != 1) & (state.agent.position.y != 0):
            actionnum = 3
            # print("cond3")
            if (relevantsamelanespeed > 2):
                actionnum = 2
        elif (len(relevantsamelanepos) > 0) & (onestephazard != 1):
            actionnum = 3
            # print("cond4")
        elif (state.agent.position.y !=0) and ( relevantspeed == 1):
            actionnum = 3
        elif (state.agent.position.y ==0) and (len(relevantsamelanepos) > 0):
            actionnum = 1
            # print("cond5")

        if state.agent.position.x == 1:
            actionnum = 4

        p = random.uniform(0.0, 1.0)
        if p > epsilon:
            with torch.no_grad():
                return self.forward(state_tensor).detach().max(1)[1][0].item()
        else:
            return actionnum



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
class AtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

def compute_loss(model, target, states, rewards, actions, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''
    # collect the q values for all the actions
    output = model(states)
    state_action_values = output.gather(1, actions)
    next_state_values = target(next_states).max(1)[0].detach().unsqueeze(1)
    next_state_values -= next_state_values * dones
    expected_state_action_values = (next_state_values * gamma) + rewards
    return F.smooth_l1_loss(state_action_values,
                            expected_state_action_values) + 0.3*F.cross_entropy(output, actions.squeeze())


def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    if episode < 600:
        epsilon = 1
        return epsilon
    else:

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * (episode-600) / epsilon_decay)
    return epsilon


def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).

    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    model = model_class(env.world.tensor_space().shape, env.action_space.n).to(device)
    target = model_class(env.world.tensor_space().shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    # env.render()

    # print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            state_tensor = np.copy(env.world.tensor_state)
            action_num = model.act(state, epsilon, episode)

            action = env.actions[action_num]
            # Apply the action to the environment
            next_state, reward, done, info = env.step(state=deepcopy(state), action=action)

            # Save transition to replay buffer
            next_state_tensor = np.copy(env.world.tensor_state)
            memory.push(Transition(state_tensor, [reward], [action_num], next_state_tensor, [done]))
            # env.render()
            state = next_state
            episode_rewards += reward
            if done:
                break
        # print(episode_rewards)
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if (episode+1) % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
                    epsilon * 100))
        if episode % 200 == 0:
            save_model(model)
            test(model, env, 100)
    return model


def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
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
    model_path = os.path.join(script_path, 'model.pt')
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def save_model(model, i=0):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    model_path = os.path.join(script_path, f'model-cars-{i}-ts-{time.time()}.pt')
    rolling_model = os.path.join(script_path, f'model.pt')
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)
    torch.save(data, rolling_model)


def get_env(cars=7, tensor=False):
    '''
    Get the sample test cases for training and testing.
    '''
    # config = {'agent_speed_range': [-3, -1], 'width': 50,
    #           'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
    #                     LaneSpec(cars=8, speed_range=[-2, -1]),
    #                     LaneSpec(cars=6, speed_range=[-1, -1]),
    #                     LaneSpec(cars=6, speed_range=[-3, -1]),
    #                     LaneSpec(cars=7, speed_range=[-2, -1]),
    #                     LaneSpec(cars=8, speed_range=[-2, -1]),
    #                     LaneSpec(cars=6, speed_range=[-3, -2]),
    #                     LaneSpec(cars=7, speed_range=[-1, -1]),
    #                     LaneSpec(cars=6, speed_range=[-2, -1]),
    #                     LaneSpec(cars=8, speed_range=[-2, -2])]
    #           }
    config = {'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=cars, speed_range=[-2, -1]),
                        LaneSpec(cars=cars, speed_range=[-2, -1]),
                        LaneSpec(cars=cars, speed_range=[-1, -1]),
                        LaneSpec(cars=cars, speed_range=[-3, -1]),
                        LaneSpec(cars=cars, speed_range=[-2, -1]),
                        LaneSpec(cars=cars, speed_range=[-2, -1]),
                        LaneSpec(cars=cars, speed_range=[-3, -2]),
                        LaneSpec(cars=cars, speed_range=[-1, -1]),
                        LaneSpec(cars=cars, speed_range=[-2, -1]),
                        LaneSpec(cars=cars, speed_range=[-2, -2])]
              }
    if tensor:
        config['observation_type'] = 'tensor'
    return gym.make('GridDriving-v0', **config)


if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    # parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    # args = parser.parse_args()

    # for i in range(0, 9, 2):
    #     print("========================= > Trying to optimise with", i, "cars...")
    env = get_env()
    model = train(AtariDQN, env)
    save_model(model)
