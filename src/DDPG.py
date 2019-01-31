"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import seaborn as sns
import matplotlib.pyplot as plt

import time

from envs.predator_prey_env import PredatorPreyEnv


#####################  hyper parameters  ####################



###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):

        self.MEMORY_CAPACITY = 10000
        self.BATCH_SIZE = 32

        self.HIDDEN_1_SIZE = 40
        self.HIDDEN_2_SIZE = 30
        self.LR_A = 1e-3    # learning rate for actor
        self.LR_C = 1e-3   # learning rate for critic
        self.GAMMA = 0.9     # reward discount
        self.TAU = 0.01      # soft replacement

        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')



        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea), tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def save_model(self, name="model"):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "saved_models/" + name + ".ckpt")
        print("Model saved in path: %s" % save_path)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            net = tf.layers.dense(s, self.HIDDEN_1_SIZE, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            net_2 = tf.layers.dense(net, self.HIDDEN_2_SIZE, activation=tf.nn.relu,
                                    kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)
            a = tf.layers.dense(net_2, self.a_dim, activation=tf.nn.tanh,
                                kernel_initializer=init_w, bias_initializer=init_b, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            hidden_1 = tf.layers.dense(s, self.HIDDEN_1_SIZE, activation=tf.nn.relu,
                                       kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
            with tf.variable_scope('hidden_2'):
                w2_s = tf.get_variable('w2_s', [self.HIDDEN_1_SIZE, self.HIDDEN_2_SIZE], initializer=init_w)
                w2_a = tf.get_variable('w2_a', [self.a_dim, self.HIDDEN_2_SIZE], initializer=init_w)
                b2 = tf.get_variable('b2', [1, self.HIDDEN_2_SIZE], initializer=init_b)
                hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2_s) + tf.matmul(a, w2_a) + b2)
            q = tf.layers.dense(hidden_2, 1, kernel_initializer=init_w, bias_initializer=init_b, name='q')
            return q

# ###############################  training  ####################################


# try:
#     env = PredatorPreyEnv(rob_type="simulation", use_torch=False, timestep=200, move_ms = 800)
#     env = env.unwrapped
#     # env.seed(1)

#     s_dim = env.observation_space.shape[0]
#     a_dim = env.action_space.shape[0]
#     a_bound = env.action_space.high

#     ddpg = DDPG(a_dim, s_dim, a_bound)

#     var = 3  # control exploration
#     episode_reward = []
#     for i in range(MAX_EPISODES+1):
#         s = env.reset()
#         ep_reward = 0
#         if var >= 0.1:
#             var *= .985  # decay the action randomness

#         done = False
#         while done is False:
#             # Add exploration noise
#             a = ddpg.choose_action(s)
#             print("choose_action:", a)
#             a = np.clip(np.random.normal(a, var), 0, 50)    # add randomness to action selection for exploration
#             s_, r, done = env.step(a)

#             ddpg.store_transition(s, a, r / 10, s_)

#             if ddpg.pointer > self.MEMORY_CAPACITY:
#                 ddpg.learn()

#             s = s_
#             ep_reward += r
#             if done:
#                 episode_reward.append(ep_reward)
#                 print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#                 break

#     end = time.clock()
#     print('Running time: %s Seconds' % (end - start))

#     sns.set(style="darkgrid")
#     plt.figure(1)
#     plt.plot(episode_reward, label='DDPG')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.legend(loc='best')

#     plt.show()

# except KeyboardInterrupt:
#     try:
#         env.close()
#     except: pass
# except Exception:
#     try:
#         env.close()
#     except: pass
#     traceback.print_exc()
