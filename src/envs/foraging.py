from gym import spaces
import robobo
import os
import time
import cv2
import numpy as np

class ForagingEnv():
    """
    Description:
        A two-wheeled robot needs to perform foraging behaviour; search and eat food
    Source:
        The Learning Machines course (2019) from the Vrije Universiteit Amsterdam.
    Observation:
        Type: Box(9)
        Num Observation
        0   Food Top Left
        1   Food Top Center
        2   Food Top Right
        3   Food Middle Left
        4   Food Middle Center
        5   Food Middle Right
        6   Food Bottom Left
        7   Food Bottom Center
        8   Food Bottom Right

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
        self.observation_space = spaces.Discrete(10)
        self.observation_labels = ["Food Top Left", "Food Top Center", "Food Top Right", "Food Middle Left", "Food Middle Center", "Food Middle Right", "Food Bottom Left", "Food Bottom Center", "Food Bottom Right", "No Food"]
        self.state = None
        self.move_ms = 250
        self.n_collected = 0

        if rob_type == "simulation":
            self.rob = robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19995)
        elif rob_type == "hardware":
            self.rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.86")
        else:
            raise Exception('rob_type should be either "simulation" or "hardware"')

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

        reward = self.get_reward()
        new_s = self.get_state()
        done = self.rob.collected_food() == 7

        self.state = new_s
        return self.state, reward, done

    def get_reward(self):
        if str(self.rob.__class__.__name__) == "SimulationRobobo":
            new_n_collected = self.rob.collected_food()
            difference = new_n_collected - self.n_collected
            self.n_collected = new_n_collected
            if difference > 0:
                print("Reward: 100")
                return 100
            else:
                print("Reward: -1")
                return -1
        else:
            return Exception("Reward function not possible on hardware")

    def mask_img(self, img):
        # Lower and upper boundary of green
        lower = np.array([0, 0, 0], np.uint8)
        upper = np.array([50, 255, 50], np.uint8)

        # Create a mask for orange
        mask = cv2.inRange(img, lower, upper)
        return mask

    def get_state(self):
        #Subimages:
        #[0 1 2
        # 3 4 5
        # 6 7 8]

        img = self.rob.get_image_front()
        # img = rgb.copy()
        # gray = self.mask_img(rgb)

        # rows_rgb, cols_rgb, channels = rgb.shape
        # rows_gray, cols_gray = gray.shape

        # rows_comb = max(rows_rgb, rows_gray)
        # cols_comb = cols_rgb + cols_gray
        # comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

        # comb[:rows_rgb, :cols_rgb] = rgb
        # comb[:rows_gray, cols_rgb:] = gray[:, :, None]

        # try:
        #     cv2.imwrite("robotview.png", comb)
        # except:
        #     pass
        greencount = []
        for i in range(3):
            for j in range(3):
                part = len(img)/3
                sub_image = img[int(part*i):int(part*(i+1)), int(part*j):int(part*(j+1))]
                sub_image = self.mask_img(sub_image)
                greencount.append(np.count_nonzero(sub_image))
        if sum(greencount) < 5:
            s = 9
        else:
            s = greencount.index(max(greencount))
        print("STATE: ", self.observation_labels[s])
        return s

    def take_super_small_movement(self):
        self.rob.move(1, 1, 500)
        time.sleep(0.2)

    def reset(self):
        try:
            self.rob.stop_world()
            time.sleep(10)
        except:
            pass
        self.rob.play_simulation()
        time.sleep(1)
        self.rob.set_phone_tilt(0.72, 100)
        self.take_super_small_movement()
        return self.get_state()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
