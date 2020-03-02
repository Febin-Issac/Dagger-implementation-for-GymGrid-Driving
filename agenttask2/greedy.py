try:
    from runner.abstracts import Agent
except:
    class Agent(object):
        pass
import random
import os
import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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
min_buffer = 1000
target_update = 20  # episode(s)
train_steps = 10
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 500
print_interval = 20


class ExampleAgent(Agent):
    '''
    An example agent that just output a random action.
    '''

    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.

        For example, you might want to load the appropriate neural networks weight
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `env` (GridDrivingEnv): the environment of the test case
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        self.env = kwargs.get('env')  # WARNING: not available in the 2nd task
        fast_downward_path = kwargs.get('fast_downward_path')
        agent_speed_range = kwargs.get('agent_speed_range')
        gamma = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('env:', env)
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def step(self, state, *args, **kwargs):
        '''
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.

        Input:
        * `state`: task 1 - `GridDrivingState`
                   task 2 -  tensor of dimension `[channel, height, width]`, with
                            `channel=[cars, agent, finish_position, occupancy_trails]` for task 2

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''
        # print('state:', state)
        liststate=[]
        relevantstate=[]
        for set in state[0]:
            liststate.append(set)
        # self.env.render()
        for cars in liststate:
            if cars.lane == state.agent.lane-1:
                relevantstate.append(cars)
        positionx =[]
        relevantspeed = relevantstate[0].speed_range[0]*-1

        for relevantcar in relevantstate:
            positionx.append(relevantcar.position.x)
            # positionx[1].append(relevantcar.speed_range[0])
        agentpos = state.agent.position.x

        relevantpos = []
        for pos in positionx:
            if (pos - agentpos >= 0 & pos - agentpos <= relevantspeed) | ((49-pos) - agentpos >= 0 & (49-pos) - agentpos <= relevantspeed):
                relevantpos.append(pos)
            # if pos - agentpos < 0 & agentpos - pos <= relevantspeed:
            action =2 #forward-1
        if (len(relevantpos) == 0) & (state.agent.position.y != 0):
            action = 0 #up
        # elif len(relevantpos) != 0:
        #     action = 3

        return action
        # return random.randrange(5)

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.

        Input:
        * `state`: task 1 - `GridDrivingState`
                   task 2 -  tensor of dimension `[channel, height, width]`, with
                            `channel=[cars, agent, finish_position, occupancy_trails]` for task 2
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state = kwargs.get('state')
        action = kwargs.get('action')
        reward = kwargs.get('reward')
        next_state = kwargs.get('next_state')
        done = kwargs.get('done')
        info = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_path, 'model.pt')


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

        if self.bufferCounter < self.bufferLimit:
            self.buffer.append(transition)
            self.bufferCounter += 1
        else:
            self.buffer[random.randint(0, self.bufferLimit - 1)] = transition

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
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

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
            action = model.act(state, epsilon)

            # print(action)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)
            env.render()
            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.001:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
                    epsilon * 100))
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
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


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
    Qvalue_current = model(states).gather(1, actions)
    Qvalue_next = target(next_states).max(1)[0].detach().unsqueeze(1)
    Qvalue_next -= Qvalue_next * dones
    expected_Qnext = Qvalue_next * gamma + rewards
    huber_Loss = F.smooth_l1_loss(Qvalue_current, expected_Qnext)
    return huber_Loss


pass


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
            return random.randint(0, self.num_actions - 1)
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


def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return ExampleAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task1_env, construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"


    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3, -1), 'gamma': 1}
            agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards) / len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards


    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])  # `test_case_id` is unique between the two task
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards) / len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break


    def get_task(task_id):
        if task_id == 1:
            test_case_id = 'task1_test'
            return {
                'time_limit': 600,
                'testcases': [{'id': test_case_id, 'env': construct_task1_env(), 'runs': 1, 't_max': 50}]
            }
        elif task_id == 2:
            tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
            return {
                'time_limit': 600,
                'testcases': [{'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max} for tc, t_max in
                              tcs]
            }
        else:
            raise NotImplementedError


    try:
        task_id = int(sys.argv[1])
    except:
        print('Run agent on an example task.')
        print('Usage: python __init__.py <task number>')
        print('Example:\n   python __init__.py 2')
        exit()

    print('Testing on Task {}'.format(task_id))

    task = get_task(2)
    timed_test(task)
