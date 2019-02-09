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

from stable_baselines.sac.sac import SAC
from stable_baselines.sac.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from envs.obstacle_avoidance import ObstacleAvoidanceEnv
from envs.foraging import ForagingEnv
from envs.predator_prey_env import PredatorPreyEnv


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

def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.1, start_epsilon=0.5, Q=None):
    if Q == None:
        Q = np.zeros([env.observation_space.n, env.action_space.n])
    else:
        print("Starting with nonzero Q values, so also setting epsilon to 0.05")
        start_epsilon = 0.05

    stats = {"episode_returns": [], "episode_lengths": []}
    Q_highscore = []
    global_steps = 0

    print("doing things with model")
    model = SAC(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50000, log_interval=10)
    print("done doing things with model")

    for i_episode in range(0,num_episodes):

        s = env.reset()
        i = 0
        R = 0
        done = False
        while done == False:
            begin_time = current_milli_time()
            eps = get_epsilon(global_steps, start_epsilon)

            # a = choose_action(env, s, Q, eps)
            a, _s = model.predict(obs)
            print("a:", a)

            #new_s, r, done = env.step(a)
            new_s, r, done, _ = env.step(a)

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
            if q_step_time < 400 or q_step_time > 600:
                print('TIME FOR WHOLE Q STEP BIGGER THAN 600: '.format(q_step_time))

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
        env = PredatorPreyEnv(rob_type, use_torch=False, timestep=200, move_ms = 800)
        if rob_type == "simulation":
            stats_multirun = []
            for i in range(5):
                Q = [[-1.15790468,-1.22759172,-1.26473357],[-1.17924785,0.14161852,-1.25618982],[-1.80221396,0.27870001,-1.36827546],[14.79871686,2.77678148,1.46590968],[25.66854012,6.15508093,3.61406269],[1.93027337,0.75870823,10.99752875],[16.70549398,32.71697651,17.5846231],[55.17954908,30.05529471,13.12186663],[11.4527288,8.45105096,35.94727484],[1.33897634,8.14143109,-0.2912973]]
                Q_q_learning, stats = q_learning(env, num_episodes=50, Q=Q, start_epsilon=0.8)
                stats_multirun.append(stats)
                print("Appended stats of run", i)
                #episode_durations = q_learning(env, 200)
                # print(episode_returns_q_learning)
            print("Final stats:", stats_multirun)
            env.close()
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
