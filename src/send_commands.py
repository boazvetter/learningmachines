#!/usr/bin/env python2
'''
V-REP commandline: ./vrep.sh -gREMOTEAPISERVERSERVICE_19995_TRUE_TRUE scenes/two_by_two_arena.ttt
Headless: add -h parameter

Send commands file for VREP
Infrared: [backR, backC, backL, frontRR, frontR,  center, frontL, frontLL]

Course: Learning Machines 2019

todolist:
- Fitness function
- Noise signal (static / dynamic : lowpass?)
- Performance increase (scene upgrades, stop world alteration(plugin removal))
- Neural net controller
- RL: policy gradient methods / direct policy search (DDPG?)

'''
from __future__ import print_function
import time
import robobo
import cv2
import os
import random
import torch
import numpy as np

COLLISIONDIST = 0.10 # Distance for collision
STATE_LABEL = ["Collision center", "Collision left", "Collision right", "No collision"]
STATES = [0, 1, 2, 3]
ACTION_LABEL = ["Driving forward", "Driving backward", "Driving left", "Driving right"]
ACTIONS = [0, 1, 2, 3]
N_EPISODES = 1000
EPISODE_LENGTH = 30
STEP_SIZE=0.05
DISCOUNT_RATE = 0.9


# def get_reward(left=0, right=0): # First get fitness at a single point in time
    # vsens = sum(rob.read_irs()) # Euclidian distance
    # return 0.5-vsens
    # left /= 100 # normalize
    # right /= 100 # normalize
    # fitness = (left+right) * (1-abs(left-right)) * (1-vsens)
    # print("Fitness: ", fitness)
    # return fitness
#todo: normalizing is not correct? normalize the outcome instead?
#todo: get_jointvelocity(self) from simulation.py instead of the current option?

def get_reward(rob, left, right):
    irs = rob.read_irs()
    frontL = irs[4] if irs[4] is not False else 1.0
    frontC = irs[5] if irs[5] is not False else 1.0
    frontR = irs[6] if irs[6] is not False else 1.0
    backR = irs[0] if irs[0] is not False else 1.0
    backC =  irs[1] if irs[1] is not False else 1.0
    backL = irs[2] if irs[2] is not False else 1.0
    # print("Sensor values:", frontL, frontC, frontR)



    collisions = (frontL < COLLISIONDIST) + (frontC < COLLISIONDIST) + (frontR < COLLISIONDIST)
    + (backR < COLLISIONDIST) + (backC < COLLISIONDIST) + (backL < COLLISIONDIST)
    # print("Collisions:", collisions)
    if collisions == 0:
        vsens = 0
    elif collisions == 1:
        vsens = 0.25
    elif collisions == 2:
        vsens = 0.5
    elif collisions > 2:
        vsens = 0.9

    # left /= 100 # normalize
    # right /= 100 # normalize
    # irs = rob.read_irs()
    # for i, value in enumerate(irs):
    #     if value is False:
    #         irs[i] = True

    # if min(irs) < COLLISIONDIST:
    #     vsens = min(irs)
    # else:
    #     vsens = 0

    reward = (left+right) * (1-abs(left-right)) * (1-vsens)
    print("reward = ", (left+right), "*", (1-abs(left-right)), "*", (1-vsens))
    return reward

def get_state(rob):
    irs = rob.read_irs()
    for i, value in enumerate(irs):
        if value is False:
            irs[i] = True

    if min(irs) < COLLISIONDIST: # near collissions
        if irs.index(min(irs)) == 5:
            state = 0
        if irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
            state = 1
        if irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
            state = 2
        else:
            state = 3 # Change this later - back IRS need to be added too
    else:
        state = 3
    return state

# def q_update(step_size=0.02, discount_rate = 0.9, epsilon = 0.1):

#     q_values = np.ones([num_states, num_actions]) * 10

#     for episode in range(0,N_EPISODES):
#         s = states["no_collision"]
#         for step in range(0,EPISODE_LENGTH):
#             a = choose_action(s)
#             r, new_s = take_action(a)
#             q[s][a] = q[s][a] + step_size * (r + discount_rate*np.argmax(q_values[new_s]) - q_values[s][a])
#             s = new_s

def choose_action(s, q_values, epsilon = 0.1):
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax(q_values[s])

def choose_random_action():
    return np.random.choice(ACTIONS)

def take_action(rob, action):
    if action == 0:
        print("Taking action 0")
        left , right = 30, 30
        #rob.move(30,30,400) # forward
    elif action == 1:
        print("Taking action 1")
        left, right = -30, -30
        #rob.move(-30,-30,400) # backward
    elif action == 2:
        print("Taking action 2")
        left, right = 0, 30
        #rob.move(0,30,400) # left
    elif action == 3:
        print("Taking action 3")
        left, right = 30, 0
        #rob.move(30,0,400) # right
    rob.move(left,right,400)
    time.sleep(0.6)

    r = get_reward(rob, left, right)
    new_s = get_state(rob)
    return r, new_s




if __name__ == "__main__":
    rob = robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19995)

    print("Playing simulation")
    rob.stop_world()
    time.sleep(10)
    rob.play_simulation()
    time.sleep(1)

    # Q-learning loop
    q_values = np.ones([len(STATES), len(ACTIONS)]) * 10.0
    for episode in range(0,N_EPISODES):
        s = 3
        for step in range(0,EPISODE_LENGTH):
            a = choose_action(s, q_values)
            r, new_s = take_action(rob, a)

            print("Reward: ", r)
            print("Old Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", q_values[s][a])
            q_values[s][a] = q_values[s][a] + (STEP_SIZE * (r + DISCOUNT_RATE*np.argmax(q_values[new_s]) - q_values[s][a]))
            print("New Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", q_values[s][a])
            s = new_s

    print("Stopping world")
    rob.stop_world()
    time.sleep(10)