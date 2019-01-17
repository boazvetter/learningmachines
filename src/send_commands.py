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
STEP_SIZE=0.5
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
        vsens = 2
    elif collisions == 2:
        vsens = 3
    elif collisions > 2:
        vsens = 5


    # irs = rob.read_irs()
    # for i, value in enumerate(irs):
    #     if value is False:
    #         irs[i] = True

    # if min(irs) < COLLISIONDIST:
    #     vsens = min(irs)
    # else:
    #     vsens = 0
    #print("left, right from reward before normalize: ", left, right)
    left /= float(100) # normalize
    right /= float(100) # normalize
    #print("left, right from reward: ", left, right)
    reward = (left+right) * (1-abs(left-right)) * (1-vsens)
    print("reward = ", (left+right), "*", (1-abs(left-right)), "*", (1-vsens))
    return reward

def get_state(rob):
    irs = rob.read_irs()
    for i, value in enumerate(irs):
        if value is False:
            irs[i] = True

    print("irs from get_state: ", irs)
    print("min irs", min(irs))
    print("irs.index(min(irs))", irs.index(min(irs)))
    if min(irs) < COLLISIONDIST: # Collissions
        if irs.index(min(irs)) == 5:
            state = 0
        elif irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
            state = 1
        elif irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
            state = 2
        elif irs.index(min(irs)) == 0 or irs.index(min(irs)) == 1 or irs.index(min(irs)) == 2:
            state = 3
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
    return state


def get_state_hardware(rob):
    irs = rob.read_irs()
    for i, value in enumerate(irs):
        if value is False:
            irs[i] = True

    print("irs from get_state: ", irs)
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
        left, right = -15, 15
        #rob.move(0,30,400) # left
    elif action == 3:
        print("Taking action 3")
        left, right = 15, -15
        #rob.move(30,0,400) # right
    rob.move(left,right,500)
    time.sleep(0.2)

    r = get_reward(rob, left, right)
    new_s = get_state_hardware(rob)
    new_s = get_state(rob)
    return r, new_s


def update_highscore():
    if return_per_episode[-1] == max(return_per_episode):
        print("NEW HIGHSCORE!")
        return True
    else:
        print("No new highscore")
        return False


if __name__ == "__main__":
    try:
        rob = robobo.SimulationRobobo(0).connect(address=os.environ.get('HOST_IP'), port=19995)
        # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.22")

        # Q-learning loop
        q_values = np.ones([len(STATES), len(ACTIONS)]) * 2.0
        return_per_episode = []
        q_values_highscore = []
        for episode in range(0,N_EPISODES):
            print("Playing simulation")
            rob.play_simulation()
            time.sleep(1)

            s = 3
            episode_return = 0
            for step in range(0,EPISODE_LENGTH):
                a = choose_action(s, q_values)
                r, new_s = take_action(rob, a)

                print("Reward: ", r)
                print("Old Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", q_values[s][a])
                q_values[s][a] = q_values[s][a] + (STEP_SIZE * (r + DISCOUNT_RATE*np.argmax(q_values[new_s]) - q_values[s][a]))
                print("New Q value for", STATE_LABEL[s], ACTION_LABEL[a], ":", q_values[s][a])
                print("Q values: \n", q_values)
                s = new_s
                episode_return += r


        # DEBUG
        # rob.move(80,100,5000)
        # print("State: ", get_state(rob), STATE_LABEL[get_state(rob)])
            return_per_episode.append(episode_return)
            print("--- Return per episode: ", return_per_episode, "---")

            if update_highscore() == True:
                q_values_highscore = copy.deepcopy(q_values)

            print("------------ q_values_highscore --------- = \n", q_values_highscore)                   

            print("Stopping world")
            rob.stop_world()
            time.sleep(10)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            rob.stop_world()
            sys.exit(0)
        except SystemExit:
            os._exit(0)


'''
tonights run
q values latest:
 [[ 1.37186483 -0.59999921  2.31960774  1.75802191]
 [ 1.90216479 -0.59999994  2.16551051  2.15707724]
 [ 1.17778567 -0.58069664  1.83677359  1.65232178]
 [ 1.18873665 -0.37358784  0.          0.        ]
 [ 1.57095874 -0.27578395  0.          0.22144549]
 [ 1.75103703 -0.29199475  0.          0.22970482]
 [ 1.02598515 -0.03        0.          0.15838875]
 [ 0.86205469 -0.30739501  0.04169621  0.        ]
 [ 0.64449579 -0.46799716  0.30664209  0.03396182]]


e-greedy e=0.1, alpha = 0.05, runtime=60, ~700 runs
q values highest:
 [[-0.03       -0.03        1.45358028  0.        ]
 [ 0.25403141  0.0315      1.04253518  0.        ]
 [-0.0135     -0.0585      1.43584427  0.0855    ]
 [ 0.78908288 -0.11129625  0.          0.        ]
 [ 0.27146288  0.          0.          0.        ]
 [ 1.22604787  0.          0.          0.15046931]
 [ 1.14847034  0.          0.          0.        ]
 [ 0.89406021 -0.0585      0.09        0.        ]
 [ 0.58095798 -0.09907866  0.50635005  0.09      ]]


BEST Q VALUES FROM YESTERDAY, :
q_values = [ [ 0.0315,-0.03,1.18698907,0.],
             [ 0.0315,-0.0585, 0.71164229,0.33388875],
             [ -0.03, -0.0585, 1.24593788, 0.09],
             [ 1.061425, -0.03,0.54435207,0.],
             [ 0.61405052,0.,1.,0.],
             [ 0.74229056,-0.0585,1.,0.],
             [ 0.79851067,-0.03,1.,0.],
             [ 0.67220085,-0.0585,1.045,0.0855],
             [ 0.64840999,-0.06242587,0.0855,0.0855]]


Optimistic, 0.1 epsilon greedy after 20 minutes of training
 [[ 1.4         0.05        1.90241699  1.86032877]
 [ 1.075       1.075       2.062854    1.8778656 ]
 [ 0.95        0.7         1.87794003  0.56022429]
 [ 0.9625      1.3953125   0.82316866  1.29725872]
 [ 2.0375      1.4125      0.725       1.175     ]
 [ 1.9         1.646875    1.55542736  1.4875    ]
 [ 1.8859375   0.63125     1.4         2.44101548]
 [ 1.76439362  0.65117188  0.59925384  2.48480225]
 [ 0.77022061  1.20341797  0.89995206  2.69439332]]
'''