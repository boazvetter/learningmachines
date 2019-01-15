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
N_EPISODES = 1
EPISODE_LENGTH = 30
STEP_SIZE=0.02
DISCOUNT_RATE = 0.9


def get_reward(left=0, right=0): # First get fitness at a single point in time
    vsens = sum(rob.read_irs()) # Euclidian distance
    return 1-vsens
    # left /= 100 # normalize
    # right /= 100 # normalize
    # fitness = (left+right) * (1-abs(left-right)) * (1-vsens)
    # print("Fitness: ", fitness)
    # return fitness
#todo: normalizing is not correct? normalize the outcome instead?
#todo: get_jointvelocity(self) from simulation.py instead of the current option?

def get_state(rob):
    # States: No colission, Near collision center, near collision right, near colission left
    ir = rob.read_irs()
    if ir[5] < COLLISIONDIST:
        state = 0
    elif ir[2] < COLLISIONDIST*1.5:
        state = 1
    elif ir[7] < COLLISIONDIST*1.5:
        state = 2
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
        rob.move(50,50,500) # forward
    elif action == 1:
        print("Taking action 1")
        rob.move(-50,-50,500) # backward
    elif action == 2:
        print("Taking action 2")
        rob.move(0,50,500) # left
    elif action == 3:
        print("Taking action 3")
        rob.move(50,0,500) # right

    time.sleep(0.5)

    r = get_reward()
    new_s = get_state(rob)
    return r, new_s




if __name__ == "__main__":
    rob = robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19995)

    try:
        print("Stopping simulation")
        rob.stop_world()
        time.sleep(10)
    except:
        pass
    print("Playing simulation")
    rob.play_simulation()
    time.sleep(1)

    # Q-learning loop
    q_values = np.ones([len(STATES), len(ACTIONS)]) * 10
    for episode in range(0,N_EPISODES):
        s = 3
        for step in range(0,EPISODE_LENGTH):
            a = choose_action(s, q_values)
            r, new_s = take_action(rob, a)
            print("Old Q value for", s, a, ":", q_values[s][a])
            q_values[s][a] = q_values[s][a] + (STEP_SIZE * (r + DISCOUNT_RATE*np.argmax(q_values[new_s]) - q_values[s][a]))
            print("New Q value for", s, a, ":", q_values[s][a])
            s = new_s

    print("Stopping world")
    rob.stop_world()
    time.sleep(10)


#discretisize states + actions
