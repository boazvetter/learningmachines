#!/usr/bin/env python2
'''
V-REP commandline: ./vrep.sh -gREMOTEAPISERVERSERVICE_19995_TRUE_TRUE scenes/two_by_two_arena.ttt
Headless: add -h parameter

Send commands file for VREP
Infrared: [backR, backC, backL, frontRR, frontR,  center, frontL, frontLL]

Course: Learning Machines 2019
By: Boaz Vetter, Jelle van Mil, Demster Bijl
'''

from __future__ import print_function
import time
import traceback
import random
import sys
import copy
import math
import random
from collections import namedtuple
from itertools import count

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from envs.obstacle_avoidance import ObstacleAvoidanceEnv
from envs.foraging import ForagingEnv
from envs.predator_prey_env import PredatorPreyEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 3) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


EPISODE_LENGTH = 300
BATCH_SIZE = 10
TARGET_UPDATE = 10


policy_net = DQN(320, 240).to(device)
target_net = DQN(320, 240).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


steps_done = 0


episode_durations = []

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





def optimize_model(discount_factor=0.99):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.float() * discount_factor) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def dqn_learning(env, num_episodes, discount_factor=0.9, alpha=0.5, epsilon=0.1):
    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset(as_tensor=True)
        state = env.get_state(as_tensor=True)
        state = state.unsqueeze(0)

        for t in count():
            # Select and perform an action
            action = select_action(state, epsilon)
            _, reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = env.get_state(as_tensor=True)
                next_state = next_state.unsqueeze(0)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(discount_factor)
            if done:
                episode_durations.append(t + 1)
                print(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return episode_durations




# For DQN net
def select_action(state, epsilon):
    sample = np.random.random()
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


# For Q learning
def choose_action(env, s, Q, epsilon = 0.1):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])

def choose_random_action(env):
    return env.action_space.sample()

def is_highscore(returns_per_episode):
    return returns_per_episode[-1] == max(returns_per_episode)

def get_epsilon(it, start=1.0):
    return max(0.05, start - it * 0.0009)

def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.1, start_epsilon=0.5, Q=None):
    if Q == None:
        Q = np.zeros([env.observation_space.n, env.action_space.n])

    stats = {"episode_returns": [], "episode_lengths": []}
    Q_highscore = []
    global_steps = 0

    for i_episode in range(0,num_episodes):

        s = env.reset()
        i = 0
        R = 0
        done = False
        while done == False:
            eps = get_epsilon(global_steps, start_epsilon)
            print("epsilon:", eps)
            #print("Epsilon:", eps)
            a = choose_action(env, s, Q, eps)
            new_s, r, done = env.step(a)

            #print("Reward: ", r)
            #print("Old Q value for", env.observation_labels[s], env.action_labels[a], ":", Q[s][a])
            Q[s][a] = Q[s][a] + (alpha * (r + discount_factor*np.max(Q[new_s]) - Q[s][a]))
            #print("New Q value for", env.observation_labels[s], env.action_labels[a], ":", Q[s][a])
            #print("Q values: \n", Q)
            s = new_s
            R += r
            i += 1
            global_steps += 1

        stats["episode_returns"].append(R)
        stats["episode_lengths"].append(i)
        print("---", stats, "---")

        # plot_Q(env, Q)

        if is_highscore(stats["episode_returns"]):
            Q_highscore = copy.deepcopy(Q)
            print("NEW BEST Q VALUES FOUND")

        print("Best Q-values until now:")
        print(Q_highscore)

    return Q_highscore, stats

def move_loop(env, Q, n=1000):
    s = env.get_state()
    for i in range(n):
        a = choose_action(env, s, Q, 0.0)
        s, _, _ = env.step(a)


# def plot_Q(env, Q):
#     fig, ax = plt.subplots()
#     im = ax.imshow(Q)

#     # We want to show all ticks...
#     ax.set_xticks(np.arange(env.observation_space.n))
#     ax.set_yticks(np.arange(env.action_space.n)
#     # ... and label them with the respective list entries
#     ax.set_xticklabels(env.observation_labels)
#     ax.set_yticklabels(env.action_labels)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     for i in range(env.action_space.n)):
#         for j in range(env.observation_space.n):
#             text = ax.text(j, i, round(Q[i][j],2),
#                            ha="center", va="center", color="w")

#     ax.set_title("Q values")
#     plt.rcParams["figure.figsize"] = (10,5)
#     fig.tight_layout()
#     plt.savefig("q_values.png")
#     # plt.close()

def main(rob_type="simulation"):
    try:
        env = PredatorPreyEnv(rob_type, use_torch=False, timestep=200, move_ms = 800)
        if rob_type == "simulation":
            #Q = [[4.45060120,-2.06225533,-1.81768766],[-1.03861586,1.80691349,-1.91895023],[2.77934013,-1.46312114,-1.84231539],[0.000982035172,0.0296677677,-0.0600853476],[8.76705569,-0.814653054,-0.0799524677],[-0.558391604,-0.548860758,4.51479662],[13.3267505,0.384440106,0.585902213],[15.0547359,0.00000000,0.143453233],[10.47699170,2.80120404,-0.0673019374],[-2.22933065,4.96889182,-2.21165651]]
            stats_multirun = []
            for i in range(5):
                Q = None
                Q_q_learning, stats = q_learning(env, 50, Q=Q)
                stats_multirun.append(stats)
                print("Appended stats of run", i)
                #episode_durations = q_learning(env, 200)
                # print(episode_returns_q_learning)
            print("Final stats:", stats_multirun)
            env.stop()
        elif rob_type == "hardware":
            print("hardware")
            # Q = [[-0.44, 0.22153309, -0.57, -0.3], [-0.3454, -0.16795115, -0.1, -0.21768462], [-0.566, -0.10283485, -0.3, -0.28], [1.23748891, -0.49995302, -0.40511995, -0.37325519], [-0.17464731640341957, 0.61231007, 3.0354045876814824, 0.046495886876921744], [1.00115432, 1.4645077162470928, 3.4647981388329692, 1.90722029], [-0.18697424, 1.38497489, 0.37807747, 3.4394753686573045], [4.2182948983594049, 0.4034608, 1.65523245, 1.48061655], [4.1249884667392127, 3.0061457889920646, 3.66441848, 3.4528335033202016]]
            Q = [[4.45060120,-2.06225533,-1.81768766],[-1.03861586,1.80691349,-1.91895023],[2.77934013,-1.46312114,-1.84231539],[0.000982035172,0.0296677677,-0.0600853476],[8.76705569,-0.814653054,-0.0799524677],[-0.558391604,-0.548860758,4.51479662],[13.3267505,0.384440106,0.585902213],[15.0547359,0.00000000,0.143453233],[10.47699170,2.80120404,-0.0673019374],[-2.22933065,4.96889182,-2.21165651]]
            move_loop(env, Q=Q, n=1000)
    except KeyboardInterrupt:
        if rob_type == "simulation":
            try:
                env.close()
            except: pass
        raise SystemExit
    except Exception:
        try:
            env.close()
        except: pass
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(rob_type=sys.argv[1])
    else:
        main()
