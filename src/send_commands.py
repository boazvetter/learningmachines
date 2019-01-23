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

import cv2
import random
import numpy as np
import sys
import copy
from envs.obstacle_avoidance import ObstacleAvoidanceEnv
from envs.foraging import ForagingEnv

EPISODE_LENGTH = 300
STEP_SIZE=0.10
DISCOUNT_RATE = 0.9

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

def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.5, epsilon=0.1, Q=None):
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
            eps = get_epsilon(global_steps, 0.5)
            print("Epsilon:", eps)
            a = choose_action(env, s, Q, eps)
            new_s, r, done = env.step(a)

            print("Reward: ", r)
            print("Old Q value for", env.observation_labels[s], env.action_labels[a], ":", Q[s][a])
            Q[s][a] = Q[s][a] + (STEP_SIZE * (r + DISCOUNT_RATE*np.max(Q[new_s]) - Q[s][a]))
            print("New Q value for", env.observation_labels[s], env.action_labels[a], ":", Q[s][a])
            print("Q values: \n", Q)
            s = new_s
            R += r
            i += 1
            global_steps += 1

        stats["episode_returns"].append(R)
        stats["episode_lengths"].append(i)
        print("---", stats, "---")

        # plot_Q(env, Q)
        print("Stopping world")
        env.rob.stop_world()
        time.sleep(5)

        if is_highscore(stats["episode_returns"]):
            Q_highscore = copy.deepcopy(Q)

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
        env = ForagingEnv(rob_type)
        if rob_type == "simulation":
            # Q = [[4.45060120,-2.06225533,-1.81768766],[-1.03861586,1.80691349,-1.91895023],[2.77934013,-1.46312114,-1.84231539],[0.000982035172,0.0296677677,-0.0600853476],[8.76705569,-0.814653054,-0.0799524677],[-0.558391604,-0.548860758,4.51479662],[13.3267505,0.384440106,0.585902213],[15.0547359,0.00000000,0.143453233],[10.47699170,2.80120404,-0.0673019374],[-2.22933065,4.96889182,-2.21165651]]
            Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning) = q_learning(env, 200)
            print(episode_returns_q_learning)
        elif rob_type == "hardware":
            print("hardware")
            # Q = [[-0.44, 0.22153309, -0.57, -0.3], [-0.3454, -0.16795115, -0.1, -0.21768462], [-0.566, -0.10283485, -0.3, -0.28], [1.23748891, -0.49995302, -0.40511995, -0.37325519], [-0.17464731640341957, 0.61231007, 3.0354045876814824, 0.046495886876921744], [1.00115432, 1.4645077162470928, 3.4647981388329692, 1.90722029], [-0.18697424, 1.38497489, 0.37807747, 3.4394753686573045], [4.2182948983594049, 0.4034608, 1.65523245, 1.48061655], [4.1249884667392127, 3.0061457889920646, 3.66441848, 3.4528335033202016]]
            Q = [[4.45060120,-2.06225533,-1.81768766],[-1.03861586,1.80691349,-1.91895023],[2.77934013,-1.46312114,-1.84231539],[0.000982035172,0.0296677677,-0.0600853476],[8.76705569,-0.814653054,-0.0799524677],[-0.558391604,-0.548860758,4.51479662],[13.3267505,0.384440106,0.585902213],[15.0547359,0.00000000,0.143453233],[10.47699170,2.80120404,-0.0673019374],[-2.22933065,4.96889182,-2.21165651]]
            move_loop(env, Q=Q, n=1000)
    except KeyboardInterrupt:
        if rob_type == "simulation":
            try: env.rob.stop_world()
            except: pass
        raise SystemExit

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(rob_type=sys.argv[1])
    else:
        main()
