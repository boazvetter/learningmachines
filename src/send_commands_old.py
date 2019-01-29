#!/usr/bin/env python2
'''
V-REP commandline: ./vrep.sh -gREMOTEAPISERVERSERVICE_19995_TRUE_TRUE scenes/two_by_two_arena.ttt
Headless: add -h parameter

Send commands file for VREP
Infrared: [backR, backC, backL, frontRR, frontR,  , frontL, frontLL]

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

current_milli_time = lambda: int(round(time.time() * 1000))

if __name__ == "__main__":

    env = PredatorPreyEnv(rob_type="simulation", use_torch=False, timestep=200, move_ms = 100)

    s = env.reset()

    time.sleep(1)

    try:
        past_time = 0
        for i in range(0,5):
            print(current_milli_time() - past_time)
            past_time = current_milli_time()
            env.step(np.random.choice([0,1,2,3]))
            # env.rob.move_continuous(30, 30, millis=100)
        env.close()
    except KeyboardInterrupt:
        try:
            env.close()
        except: pass
        raise SystemExit
    except Exception:
        try:
            env.close()
        except: pass
        traceback.print_exc()

    env.close()

