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
from env import construct_task1_env, construct_task2_env
from mcts import MonteCarloTreeSearch, GridWorldState, randomPolicy
from copy import deepcopy


from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma = 0.98
buffer_limit = 10000
batch_size = 32
max_episodes = 2000
t_max = 600
min_buffer = 1000
target_update = 20  # episode(s)
train_steps = 10
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 500
print_interval = 20
glob_count = 0

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.bufferCounter = 0
        self.buffer = []
        self.bufferLimit = buffer_limit

        pass

    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.
        Output:
            * None
            '''

        if self.bufferCounter < self.bufferLimit :
             self.buffer.append(transition)
             self.bufferCounter += 1
        else:
            # randompos = random.randint(0, self.bufferLimit - 1)
            self.buffer[self.bufferCounter % self.bufferLimit] = transition
            self.bufferCounter += 1


    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''

        batch_data = random.sample(self.buffer, batch_size)
        transition_batch = Transition(*zip(*batch_data))
        states = torch.tensor(transition_batch[0], dtype=torch.float, device=device)
        actions = torch.tensor(transition_batch[1], dtype=torch.long, device=device)
        rewards = torch.tensor(transition_batch[2], dtype=torch.float, device=device)
        next_states = torch.tensor(transition_batch[3], dtype=torch.float, device=device)
        dones = torch.tensor(transition_batch[4], dtype=torch.float, device=device)

        return states, actions, rewards, next_states, dones

        pass

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class Base(nn.Module):
    '''
    Base neural network model that handles dynamic architecture.
    '''

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


class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if random.uniform(0, 1) < epsilon:
            return random.randint(0,self.num_actions-1)
        else:
            with torch.no_grad():
                return int(self.forward(state).max(1)[1])

    '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
    pass


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


def compute_loss(model, target, states, actions, rewards, next_states, dones):
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
    output = model(states)
    Qvalue_current = output.gather(1, actions)
    Qvalue_next = target(next_states).max(1)[0].detach().unsqueeze(1)
    Qvalue_next -= Qvalue_next * dones
    expected_Qnext = Qvalue_next * gamma + rewards
    huber_Loss = F.smooth_l1_loss(Qvalue_current, expected_Qnext)
    # CE_loss =  F.cross_entropy(output, actions.squeeze())
    # arg = 1
    # if glob_count > 400:
    #     arg = 0

    return  huber_Loss


pass


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
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def increment():
    global glob_count
    glob_count = glob_count+1

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

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    numiters = 15
    explorationParam = 1.
    random_seed = 10
    # mcts = MonteCarloTreeSearch(env=env, numiters=numiters, explorationParam=1., random_seed=random_seed)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            # state = GridWorldState(state, is_done=env.done)
            state_tensor = np.copy(env.world.tensor_state)

            # print('state:', state)
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
                if ((pos - agentpos >= 0 )& (pos - agentpos <= relevantspeed)) | (((49 + pos) - agentpos >= 0 )& ((49 + pos) - agentpos <= relevantspeed)):
                    relevantpos.append(pos)
                # if pos - agentpos < 0 & agentpos - pos <= relevantspeed:

            backtrack=[]
            relevantsamelane=[]

            for cars in liststate:
                if cars.lane == state.agent.lane:
                    relevantsamelane.append(cars)
            samelaneposx =[]

            for relevantcarsamelane in relevantsamelane:
                samelaneposx.append(relevantcarsamelane.position.x)
                # positionx[1].append(relevantcar.speed_range[0])
            # agentpos = state.agent.position.x

            relevantsamelanepos = []
            relevantsamelanespeed = relevantsamelane[0].speed_range[0] * -1

            onestephazard = 0
            frontline = []
            for pos2 in samelaneposx:
                if ((pos2 - agentpos >= 0 )& (pos2 - agentpos <= relevantsamelanespeed)) | (((49 + pos2) - agentpos >= 0 )& ((49 + pos2) - agentpos <= relevantsamelanespeed)):
                    relevantsamelanepos.append(pos2)
                if ((pos2 - agentpos) < 0) & ((agentpos- pos2) <=2):
                    onestephazard =1

            actionnum = 4  # forward-1
            if (state.agent.position.y == 0) & (onestephazard == 0):
                actionnum = 3
                # print("cond1")
            elif (len(relevantpos) == 0) & (state.agent.position.y != 0):
                actionnum = 0  # up
                # print("cond2")
            elif (len(relevantsamelanepos) > 0) & (onestephazard !=1) & (state.agent.position.y != 0):
                actionnum = 3
                # print("cond3")
                if(relevantsamelanespeed > 2):
                    actionnum = 2
            elif (len(relevantsamelanepos) > 0) & (onestephazard !=1):
                actionnum = 3
                # print("cond4")
            elif (relevantspeed == 1):
                actionnum = 3
                # print("cond5")

            if state.agent.position.x == 1:
                actionnum = 4
            # elif len(relevantpos) != 0:
            #     action = 3

            # action = mcts.buildTreeAndReturnBestAction(initialState=state)
            # print(actionnum)

            action = env.actions[actionnum]
            # done = env.step(state=deepcopy(state.state), action=action)[2]
            # action = torch.from_numpy(action).float().unsqueeze(0).to(device)

            # action = model.act(state, epsilon)

            # print(action)
            # env.render()
            # Apply the action to the environment
            next_state, reward, done, info = env.step(state=deepcopy(state), action=action)
            # env.render()
            # Save transition to replay buffer
            next_state_tensor = np.copy(env.world.tensor_state)
            memory.push(Transition(state_tensor, [env.actions.index(action)], [reward], next_state_tensor, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                # print("episode done"+ str(episode_rewards))
                break
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.0001:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())
        # increment()

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
                    epsilon * 100))
            if ((episode) % 200) == 0:
                save_model(model)
                model = get_model()
                test(model, construct_task2_env(), max_episodes=10)
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
            action = model.act(env.world.tensor_state)
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
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()


    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    # timed_test(task)
    # env = get_env()

    if args.train:
        model = train(ConvDQN, construct_task2_env())
        save_model(model)
    else:
        model = get_model()
    test(model, construct_task2_env(), max_episodes=10)