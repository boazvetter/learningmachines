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
import robobo
import cv2
import os
import random
import numpy as np
import sys
import copy

COLLISIONDIST = 0.075 # Distance for collision
STATE_LABEL = ["Collision center", "Collision right", "Collision left", "Collision back", "Near Collision Center", "Near Collision Right", "Near Collision Left", "Near Collision Back", "No collision"]
STATES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ACTION_LABEL = ["Driving forward", "Driving backward", "Driving left", "Driving right"]
ACTIONS = [0, 1, 2, 3]
N_EPISODES = 1000
EPISODE_LENGTH = 60
STEP_SIZE=0.10
DISCOUNT_RATE = 0.9

def get_reward_simple(rob, left, right):
    irs = rob.read_irs()
    irs = [ ir < COLLISIONDIST if ir != False else 0.0 for ir in irs ]
    collisions = sum(irs)
    left /= float(100) # normalize
    right /= float(100) # normalize
    reward = (left+right) - collisions
    return reward

def get_reward(rob, left, right):
    irs = rob.read_irs()
    frontL = irs[4] if irs[4] is not False else 1.0
    frontC = irs[5] if irs[5] is not False else 1.0
    frontR = irs[6] if irs[6] is not False else 1.0
    backR = irs[0] if irs[0] is not False else 1.0
    backC =  irs[1] if irs[1] is not False else 1.0
    backL = irs[2] if irs[2] is not False else 1.0

    collisions = (frontL < COLLISIONDIST) + (frontC < COLLISIONDIST) + (frontR < COLLISIONDIST)
    + (backR < COLLISIONDIST) + (backC < COLLISIONDIST) + (backL < COLLISIONDIST)
    # print("Collisions:", collisions)
    if collisions == 0:
        vsens = 0
    elif collisions == 1:
        vsens = 1.5
    elif collisions == 2:
        vsens = 3
    elif collisions > 2:
        vsens = 5

    left /= float(100) # normalize
    right /= float(100) # normalize
    move_reward = (0.8 * (left+right)) + (left+right) + 0.1
    reward = move_reward * (1-abs(left-right)) * (1-vsens)
    print("reward = ", move_reward, "*", (1-abs(left-right)), "*", (1-vsens), "=", reward)
    return reward

def get_state(rob):
    irs = rob.read_irs()
    for i, value in enumerate(irs):
        if value is False:
            irs[i] = True

    if min(irs) < COLLISIONDIST: # Collissions
        if irs.index(min(irs)) == 5:
            state = 0
        elif irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
            state = 1
        elif irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
            state = 2
        elif irs.index(min(irs)) == 0 or irs.index(min(irs)) == 1 or irs.index(min(irs)) == 2:
            state = 3
        print(STATE_LABEL[state])
        return state
    elif min(irs) < COLLISIONDIST*2: # Near collissions
        if irs.index(min(irs)) == 5:
            state = 4
        elif irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
            state = 5
        elif irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
            state = 6
        elif irs.index(min(irs)) == 0 or irs.index(min(irs)) == 1 or irs.index(min(irs)) == 2:
            state = 7
    else:
        state = 8
    print(STATE_LABEL[state])
    return state


def get_state_hardware(rob):
    irs = rob.read_irs()
    for i, value in enumerate(irs):
        if value is False:
            irs[i] = True

    #print("irs from get_state: ", irs)
    # print("min irs", max(irs))
    # print("irs.index(max(irs))", irs.index(max(irs)))
    if max(irs) > 200: # Collissions
        if irs.index(max(irs)) == 5:
            state = 0
        elif irs.index(max(irs)) == 3 or irs.index(max(irs)) == 4:
            state = 1
        elif irs.index(max(irs)) == 6 or irs.index(max(irs)) == 7:
            state = 2
        elif irs.index(max(irs)) == 0 or irs.index(max(irs)) == 1 or irs.index(max(irs)) == 2:
            state = 3
        print(STATE_LABEL[state])
        return state
    elif max(irs) > 60: # Near collissions
        if irs.index(max(irs)) == 5:
            state = 4
        elif irs.index(max(irs)) == 3 or irs.index(max(irs)) == 4:
            state = 5
        elif irs.index(max(irs)) == 6 or irs.index(max(irs)) == 7:
            state = 6
        elif irs.index(max(irs)) == 0 or irs.index(max(irs)) == 1 or irs.index(max(irs)) == 2:
            state = 7
        print(STATE_LABEL[state])
    else:
        state = 8
        print(STATE_LABEL[state])

    return state

def choose_action(s, Q, epsilon = 0.1):
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax(Q[s])

def choose_random_action():
    return np.random.choice(ACTIONS)

def take_action(rob, action):
    if action == 0:
        print("Taking action 0", ACTION_LABEL[0])
        left , right = 30, 30
        #rob.move(30,30,400) # forward
    elif action == 1:
        print("Taking action 1", ACTION_LABEL[1])
        left, right = -30, -30
        #rob.move(-30,-30,400) # backward
    elif action == 2:
        print("Taking action 2", ACTION_LABEL[2])
        left, right = -15, 15
        #rob.move(0,30,400) # left
    elif action == 3:
        print("Taking action 3", ACTION_LABEL[3])
        left, right = 15, -15
        #rob.move(30,0,400) # right
    rob.move(left,right,500)
    time.sleep(0.2)

    r = get_reward(rob, left, right)
    new_s = get_state(rob)
    done = new_s < 4
    return r, new_s, done


def is_highscore(returns_per_episode):
    return returns_per_episode[-1] == max(returns_per_episode)


def q_learning(rob, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, Q=None):
    # Q-learning loop
    Q = np.zeros([len(STATES), len(ACTIONS)])
    stats = {"episode_returns": [], "episode_lengths": []}
    Q_highscore = []

    for i_episode in range(0,N_EPISODES):
        print("Playing simulation")
        rob.play_simulation()
        time.sleep(1)

        s = get_state(rob)

        i = 0
        R = 0
        done = False
        while done == False:
            a = choose_action(s, Q, 0.1)
            r, new_s, done = take_action(rob, a)

            print("Reward: ", r)
            print("Old Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", Q[s][a])
            Q[s][a] = Q[s][a] + (STEP_SIZE * (r + DISCOUNT_RATE*np.max(Q[new_s]) - Q[s][a]))
            print("New Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", Q[s][a])
            print("Q values: \n", Q)
            s = new_s
            R += r
            i += 1

        stats["episode_returns"].append(R)
        stats["episode_lengths"].append(i)
        print("---", stats, "---")
        print("Stopping world")
        rob.stop_world()
        time.sleep(5)

        if is_highscore(stats["episode_returns"]):
            Q_highscore = copy.deepcopy(Q)

    return Q_highscore, stats

def main():
    try:
        rob = robobo.SimulationRobobo(0).connect(address=os.environ.get('HOST_IP'), port=19997)
        # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.22")
        Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning) = q_learning(rob, 20)
        print(episode_returns_q_learning)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            rob.stop_world()
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    main()
