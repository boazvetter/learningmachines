from gym import spaces
import robobo
import os
import time

class ObstacleAvoidanceEnv():
    """
    Description:
        A two-wheeled robot has the task to move and avoid obstacles on the way.
    Source:
        The Learning Machines course (2019) from the Vrije Universiteit Amsterdam.
    Observation:
        Type: Box(9)
        Num Observation
        0   Collision center
        1   Collision right
        2   Collision left
        3   Collision back
        4   Near collision center
        5   Near collision right
        6   Near collision left
        7   Near collision back
        8   No collision

    Actions:
        Type: Discrete(4)
        Num Actions             Speed left      Speed right
        0   Driving forward     30              30
        1   Driving backward    -30             -30
        2   Driving left        -15             15
        3   Driving right       15              -15

    Reward:
        Reward is dependent on the forward movement and the collision state after the movement
    Starting State:
        No collision
    Episode Termination:
        After 60 episodes
    """

    def __init__(self, rob_type):
        self.action_space = spaces.Discrete(4)
        self.action_labels = ["Driving forward", "Driving backward", "Driving left", "Driving right"]
        self.observation_space = spaces.Discrete(9)
        self.observation_labels = ["Collision center", "Collision right", "Collision left", "Collision back", "Near Collision Center", "Near Collision Right", "Near Collision Left", "Near Collision Back", "No collision"]
        self.state = None
        self.move_ms = 250
        self.sim_collision_threshold = 0.0375
        self.sim__near_collision_threshold = 0.15
        self.hardware_collision_threshold = 200
        self.hardware_near_collision_threshold = 40

        if rob_type == "simulation":
            self.rob = robobo.SimulationRobobo(0).connect(address=os.environ.get('HOST_IP'), port=19995)
        elif rob_type == "hardware":
            self.rob = robobo.HardwareRobobo(camera=False).connect(address="192.168.1.86")
        else:
            Exception()



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        if action == 0:
            left , right = 30, 30
        elif action == 1:
            left, right = -30, -30
        elif action == 2:
            left, right = -15, 15
        elif action == 3:
            left, right = 15, -15
        self.rob.move(left,right,self.move_ms)
        time.sleep(0.2)

        reward = self.get_reward(left, right)
        new_s = self.get_state()
        done = new_s < 4

        self.state = new_s
        return self.state, reward, done

    def get_reward(self, left, right):
        irs = self.rob.read_irs()
        if str(self.rob.__class__.__name__) == "SimulationRobobo":
            irs = [ ir < self.sim_collision_threshold if ir != False else 0.0 for ir in irs ]
        if str(self.rob.__class__.__name__) == "HardwareRobobo":
            irs = [ ir > self.hardware_collision_threshold if ir != False else 0.0 for ir in irs ]

        collisions = sum(irs)
        left /= float(100)
        right /= float(100)
        reward = (left+right) - collisions
        return reward

    def get_state(self):
        irs = self.rob.read_irs()
        for i, value in enumerate(irs):
            if value is False:
                irs[i] = True

        if str(self.rob.__class__.__name__) == "SimulationRobobo":
            if min(irs) < self.sim_collision_threshold:
                if irs.index(min(irs)) == 5:
                    state = 0
                elif irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
                    state = 1
                elif irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
                    state = 2
                elif irs.index(min(irs)) == 0 or irs.index(min(irs)) == 1 or irs.index(min(irs)) == 2:
                    state = 3
                return state
            elif min(irs) < self.sim__near_collision_threshold:
                if irs.index(min(irs)) == 5:
                    state = 4
                elif irs.index(min(irs)) == 3 or irs.index(min(irs)) == 4:
                    state = 5
                elif irs.index(min(irs)) == 6 or irs.index(min(irs)) == 7:
                    state = 6
                elif irs.index(min(irs)) == 0 or irs.index(min(irs)) == 1 or irs.index(min(irs)) == 2:
                    state = 7
                return state
            else:
                state = 8
                return state
        elif str(self.rob.__class__.__name__) == "HardwareRobobo":
            if max(irs) > self.hardware_collision_threshold:
                if irs.index(max(irs)) == 5:
                    state = 0
                elif irs.index(max(irs)) == 3 or irs.index(max(irs)) == 4:
                    state = 1
                elif irs.index(max(irs)) == 6 or irs.index(max(irs)) == 7:
                    state = 2
                elif irs.index(max(irs)) == 0 or irs.index(max(irs)) == 1 or irs.index(max(irs)) == 2:
                    state = 3
                return state
            elif max(irs) > self.hardware_near_collision_threshold:
                if irs.index(max(irs)) == 5:
                    state = 4
                elif irs.index(max(irs)) == 3 or irs.index(max(irs)) == 4:
                    state = 5
                elif irs.index(max(irs)) == 6 or irs.index(max(irs)) == 7:
                    state = 6
                elif irs.index(max(irs)) == 0 or irs.index(max(irs)) == 1 or irs.index(max(irs)) == 2:
                    state = 7
                return state
            else:
                state = 8
        else:
            return Exception()

    def take_super_small_movement(self):
        self.rob.move(1, 1, 100)
        time.sleep(0.2)

    def reset(self):
        try:
            self.rob.stop_world()
            time.sleep(10)
        except:
            pass
        self.rob.play_simulation()
        time.sleep(1)
        self.take_super_small_movement()
        return self.get_state()



    # def get_reward_complex(left, right):
    #     irs = self.rob.read_irs()
    #     frontL = irs[4] if irs[4] is not False else 1.0
    #     frontC = irs[5] if irs[5] is not False else 1.0
    #     frontR = irs[6] if irs[6] is not False else 1.0
    #     backR = irs[0] if irs[0] is not False else 1.0
    #     backC =  irs[1] if irs[1] is not False else 1.0
    #     backL = irs[2] if irs[2] is not False else 1.0

    #     collisions = (frontL < COLLISIONDIST) + (frontC < COLLISIONDIST) + (frontR < COLLISIONDIST)
    #     + (backR < COLLISIONDIST) + (backC < COLLISIONDIST) + (backL < COLLISIONDIST)
    #     if collisions == 0:
    #         vsens = 0
    #     elif collisions == 1:
    #         vsens = 1.5
    #     elif collisions == 2:
    #         vsens = 3
    #     elif collisions > 2:
    #         vsens = 5

    #     left /= float(100) # normalize
    #     right /= float(100) # normalize
    #     move_reward = (0.8 * (left+right)) + (left+right) + 0.1
    #     reward = move_reward * (1-abs(left-right)) * (1-vsens)
    #     print("reward = ", move_reward, "*", (1-abs(left-right)), "*", (1-vsens), "=", reward)
    #     return reward
