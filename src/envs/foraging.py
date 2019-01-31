from gym import spaces
import robobo
import os
import time
import cv2
import numpy as np
import torch

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
        9   No food

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

    def __init__(self, rob_type, use_torch=False, timestep=100):
        self.action_space = spaces.Discrete(3)
        self.action_labels = ["Driving forward", "Driving left", "Driving right"]
        self.observation_space = spaces.Discrete(12)
        self.observation_labels = ["Top Left", "Top Center", "Top Right", "Middle Left", "Middle Center", "Middle Right", "Bottom Left", "Bottom Center", "Bottom Right", "No Food", "Prey was left", "Prey was Right"]
        self.state = None
        self.move_ms = 500 * 100.0/timestep
        self.n_collected = 0
        self.step_i = 0
        self.rob_type = rob_type
        self.use_torch = use_torch

        if rob_type == "simulation":
            self.rob = robobo.SimulationRobobo().connect(address=os.environ.get('HOST_IP'), port=19997)
        elif rob_type == "hardware":
            print("connecting with hardware")
            self.rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.86")
            self.rob.talk("Tilting")
            self.rob.set_phone_tilt(106, 50)
            self.rob.set_phone_tilt(109, 5)
            time.sleep(8)

        else:
            raise Exception('rob_type should be either "simulation" or "hardware"')



    def step(self, action, as_tensor=False):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        if action == 0:
            left , right = 30, 30
            self.rob.move(left,right,400)
        elif action == 1:
            left, right = -15, 15
            self.rob.move(left,right,self.move_ms)
        elif action == 2:
            left, right = 15, -15
            self.rob.move(left,right,self.move_ms)

        time.sleep(0.2)

        if str(self.rob.__class__.__name__) == "SimulationRobobo":
            reward = self.get_reward()
        else:
            reward = -1
        new_s = self.get_state()
        #print("collected_food", self.n_collected)
        print("episode step:", self.step_i)
        if self.n_collected > 17 or self.step_i > 99:
            self.step_i = 0
            done = True
        else:
            done = False
        #print("collected_food done")

        self.step_i += 1
        self.state = new_s
        return self.state, reward, done

    def get_reward(self):
        if str(self.rob.__class__.__name__) == "SimulationRobobo":
            try:
                new_n_collected = self.rob.collected_food()
                difference = new_n_collected - self.n_collected
                self.n_collected = new_n_collected
                if difference > 0:
                    #print("Reward: 10")
                    return 10*difference
                else:
                    #print("Reward: -1")
                    return -1
            except:
                return 0
        else:
            return Exception("Reward function not possible on hardware")

    def mask_img(self, img):
        if self.rob_type == "simulation":
            # Lower and upper boundary of green
            lower = np.array([0, 0, 1], np.uint8)
            upper = np.array([50, 50, 255], np.uint8)

            # Create a mask for orange
            mask = cv2.inRange(img, lower, upper)

        if self.rob_type == "hardware":
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            # cv2.imwrite('robotviews/robotview{}-hsv.jpg'.format(testtime), hsv)
            # time.sleep(0.1)
            minHSV = np.array([42, 70, 20])
            maxHSV = np.array([97, 255, 255])
            mask = cv2.inRange(hsv,minHSV,maxHSV)

            # cv2.imwrite('robotviews/robotview{}-mask.jpg'.format(testtime), mask)
            # time.sleep(0.1)
        return mask

    def get_state(self):
        if self.use_torch:
            return self.get_state_torch()
        else:
            return self.get_state_greencount()

    def get_state_greencount(self):
        #Subimages:
        #[0 1 2
        # 3 4 5
        # 6 7 8]

        img = self.rob.get_image_front()
        # img = rgb.copy()
        # gray = self.mask_img(rgb)
        if str(self.rob.__class__.__name__) == "HardwareRobobo":
            img = cv2.resize(img,(240,320))
            img = cv2.GaussianBlur(img, (9, 9), 0)



        # try:
        #     cv2.imwrite("robotview.png", img)
        #     masked = self.mask_img(img)
        #     cv2.imwrite("robotviewmasked.png", masked)
        # except:
        #     pass
        
        greencount = []
        for i in range(3):
            for j in range(3):
                part_x = img.shape[0]/3
                part_y = img.shape[1]/3
                sub_image = img[int(part_x*i):int(part_x*(i+1)), int(part_y*j):int(part_y*(j+1))]
                sub_image = self.mask_img(sub_image)
                greencount.append(np.count_nonzero(sub_image))
        if max(greencount) < 20:
            s = 9
        else:
            s = greencount.index(max(greencount))

        if s == 9 and self.state is not None:
            if self.state == 2 or self.state == 5 or self.state == 8:
                s = 11
            elif self.state == 0 or self.state == 3 or self.state == 6:
                s = 10

        #print("greencount", greencount)
        print("STATE: ", self.observation_labels[s])
        if str(self.rob.__class__.__name__) == "HardwareRobobo" and s < 9:
            self.rob.talk(self.observation_labels[s])
        return s

    def get_state_torch(self):
        img = self.rob.get_image_front()
        img = self.mask_img(img)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img)
        return img

    def get_combined_image(self, image1, image2):
        rows_image1, cols_image1, channels = image1.shape
        rows_image2, cols_image2 = image2.shape

        rows_comb = max(rows_image1, rows_image2)
        cols_comb = cols_image1 + cols_image2
        comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

        comb[:rows_image1, :cols_image1] = image1
        comb[:rows_image2, cols_image2:] = image2[:, :, None]
        return comb

    def take_super_small_movement(self):
        self.rob.move(1, 1, 500)
        time.sleep(0.2)

    def reset(self, as_tensor=False):
        try:
            self.rob.stop_world()
            time.sleep(10)
        except:
            pass

        for i in range(0,16):
            print("Setting position")
            self.rob.set_food_position(i)

        self.rob.play_simulation()
        time.sleep(1)
        self.rob.set_phone_tilt(0.72, 100)
        self.take_super_small_movement()
        return self.get_state()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
