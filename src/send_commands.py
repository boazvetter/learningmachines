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

from envs.obstacle_avoidance import ObstacleAvoidanceEnv
from envs.predator_prey_env import PredatorPreyEnv
from envs.foraging_env_box import ForagingEnvBox
from DDPG import DDPG

EPISODE_LENGTH = 300
BATCH_SIZE = 10
TARGET_UPDATE = 10


steps_done = 0


episode_durations = []



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
    return max(0.05, start - it * 0.00045)

def current_milli_time():
    return int(round(time.time() * 1000))

def ddpg_algorithm(env, num_episodes, start_variance=50, run=0):
    env = env.unwrapped
    # env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = start_variance  # control exploration

    stats = {"episode_returns": [], "episode_lengths": []}
    highscore_parameters = []
    global_steps = 0

    for i_episode in range(0,num_episodes):
        if i_episode % 10 == 0:
            s = env.reset(reset_positions=True)
        else:
            s = env.reset(reset_positions=False)
        R = 0
        i = 0
        done = False
        while done == False:
            begin_time = current_milli_time()
            if var >= 1:
                var *= .9992  # decay the action randomness

            a = ddpg.choose_action(s)

            if np.random.random() < 0.01:
                print("predicted action: ", a)

            a = np.clip(np.random.normal(a, var), 5, 25)    # add randomness to action selection for exploration
            try:
                new_s, r, done = env.step(a)
            except:
                print("error in env.step")
                continue

            ddpg.store_transition(s, a, r / 10, new_s)

            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                ddpg.learn()

            s = new_s
            R += r
            global_steps += 1
            i += 1

            q_step_time = current_milli_time() - begin_time


        stats["episode_returns"].append(R)
        stats["episode_lengths"].append(i)
        print("---", stats, "---")

        if is_highscore(stats["episode_returns"]):
            ddpg.save_model("best" + str(run))

        ddpg.save_model("last" + str(run))

    return stats



def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.1, start_epsilon=0.5, Q=None):
    if Q == None:
        Q = np.zeros([env.observation_space.n, env.action_space.n])
    else:
        print("Starting with nonzero Q values, so also setting epsilon to 0.05")
        start_epsilon = 0.05

    stats = {"episode_returns": [], "episode_lengths": []}
    Q_highscore = []
    global_steps = 0

    for i_episode in range(0,num_episodes):

        s = env.reset()
        R = 0
        i = 0
        done = False
        while done == False:
            begin_time = current_milli_time()
            eps = get_epsilon(global_steps, start_epsilon)

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

            q_step_time = current_milli_time() - begin_time
            # if q_step_time < 400 or q_step_time > 600:
            #     print('TIME FOR WHOLE Q STEP BIGGER THAN 600: '.format(q_step_time))

        stats["episode_returns"].append(R)
        stats["episode_lengths"].append(i)
        print("---", stats, "---")

        # plot_Q(env, Q)

        if is_highscore(stats["episode_returns"]):
            Q_highscore = copy.deepcopy(Q)
            print("NEW BEST Q VALUES FOUND")

        print("Best Q-values until now:")
        print(Q_highscore)

        print("\nLast Q-values:")
        print(Q)

    return Q_highscore, stats

def move_loop(env, Q, n=1000):
    s = env.get_state()
    for i in range(n):
        a = choose_action(env, s, Q, 0.0)
        s, _, _ = env.step(a)


def main(rob_type="simulation"):
    try:
        env = ForagingEnvBox(rob_type, use_torch=False, timestep=200)
        if rob_type == "simulation":
            stats_multirun = []
            for i in range(5):
                print("---------- RUN: ", i, "--------")
                stats = ddpg_algorithm(env, num_episodes=50, run=0)
                stats_multirun.append(stats)
                print("Appended stats of run", i)
                print("Final stats:", stats_multirun)
            print("Final stats:", stats_multirun)
            env.close()
        elif rob_type == "hardware":
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
