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
import robobo
import cv2
import os
import random
import torch

def bumpintowall():
    rob.move(100,100,4000)

def getFitness(left=0, right=0): # First get fitness at a single point in time
    vsens = sum(rob.read_irs()) # Euclidian distance 
    left /= 100 # normalize
    right /= 100 # normalize
    fitness = (left+right) * (1-abs(left-right)) * (1-vsens) 
    print("Fitness: ", fitness)
    return fitness
#todo: normalizing is not correct? normalize the outcome instead?
#todo: get_jointvelocity(self) from simulation.py instead of the current option?

if __name__ == "__main__":
    robs = [
        #robobo.HardwareRobobo(camera=True).connect(address="192.168.178.80"),
        robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19995),
    ]


    # robs[0].pause_simulation()
    # move and talk
    for i, rob in enumerate(robs):
        print("Playing simulation")
        rob.play_simulation()
        time.sleep(1)
        print('sending commands to robot {}'.format(i))
        getFitness()
        left = 100
        right = 30
        rob.move(left,right,2000)
        getFitness(left, right)
        #print("get joint velocity: ", rob.get_jointvelocity())
        # for i in range(10):
        #     print(i)
        #     ir = rob.read_irs()
        #     print("Infrared: ", ir)
        #     if all(ir) == False:
        #         rob.move(25,25, 200)
        #     elif ir[3] > ir[7]:
        #         rob.move(-50, -50, 500)
        #         rob.move(50, 0, 800)
        #     elif ir[3] < ir[7]:
        #         rob.move(-50, -50, 500)
        #         rob.move(0, 50, 800)


        print("Infrared values: ", rob.read_irs())
        print("Stopping world")
        rob.stop_world()
        time.sleep(10)

        # print("robobo is at {}".format(rob.position()))
        # rob.move(10, 10, 2000)
        # print("robobo is at {}".format(rob.position()))

        ## Following code moves the phone stand
        # rob.set_phone_pan(343, 100)
        # rob.set_phone_tilt(109, 100)
        # time.sleep(1)
        # rob.set_phone_pan(11, 100)
        # rob.set_phone_tilt(26, 100)

        # rob.talk('Hi, my name is Robobo {}'.format(i))
        # rob.sleep(1)
        # rob.set_emotion('happy')

        # # Following code gets an image from the camera
        # image = rob.get_image_front()
        # cv2.imwrite("test_pictures_{}.png".format(i),image)

    #     time.sleep(0.1)

    # # IR reading
    # for i in range(10):
    #     for i, rob in enumerate(robs):
    #         print("ROB {} Irs: {}".format(i, rob.read_irs()))
    #     time.sleep(1)

    # move back
    # for rob in robs:
    #     rob.move(-5, 5, 1000)

    # Stopping the simualtion resets the environment


