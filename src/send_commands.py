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

COLLISIONDIST = 0.10 # Distance for collision

def getFitness(left=0, right=0): # First get fitness at a single point in time
    vsens = sum(rob.read_irs()) # Euclidian distance 
    left /= 100 # normalize
    right /= 100 # normalize
    fitness = (left+right) * (1-abs(left-right)) * (1-vsens) 
    print("Fitness: ", fitness)
    return fitness
#todo: normalizing is not correct? normalize the outcome instead?
#todo: get_jointvelocity(self) from simulation.py instead of the current option?

def getState():
    # States: No colission, Near collision center, near collision right, near colission left
    ir = rob.read_irs()
    if ir[5] < COLLISIONDIST:
        state = "Collision center"
    elif ir[2] < COLLISIONDIST*1.5:
        state = "Collision left"
    elif ir[7] < COLLISIONDIST*1.5:
        state = "Collision right"
    else:
        state = "No Colission"

    return state

def setAction(action):
    # Actions: Drive forward, backward, left, right
    print("starting")
    if action == "forward":
        print("Driving forward")
        rob.move(50,50,500)
    elif action == "backward":
        print("Driving backward")
        rob.move(-50,-50,500)
    elif action == "left":
        print("Driving left")
        rob.move(0,50,500)
        #rob.move(50,50,500)
    elif action == "right":        
        print("Driving right")
        rob.move(50,0,500)
        #rob.move(50,50,500)

    print("done")

if __name__ == "__main__":
    

    robs = [
        #robobo.HardwareRobobo(camera=True).connect(address="192.168.178.80"),
        robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19995),
    ]

    # Main loop
    for i, rob in enumerate(robs):
        print("Playing simulation")
        rob.play_simulation()
        time.sleep(1)
        
        for i in range (1):
            setAction("forward")
            print(getState())
            setAction("forward")
            print(getState())
            setAction("forward")
            print(getState())
            setAction("forward")
            print(getState())
            setAction("forward")
            print(getState())
            print("Infrared values: ", rob.read_irs())

        print("Stopping world")
        rob.stop_world()
        time.sleep(10)


#discretisize states + actions