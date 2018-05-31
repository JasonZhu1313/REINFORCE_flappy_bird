from __future__ import print_function
import os, sys

import matplotlib.pyplot as plt

import numpy as np
from collections import deque
import random
import os, sys
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

PRO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PRO_PATH + "/game/")
sys.path.append(PRO_PATH)
import wrapped_flappy_bird as game

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import tensorflow as tf


ACTION_PER_FRAME = 1
IMAGE_ROWS, IMAGE_COLS = 80, 80
IMAGE_CHANNELS = 4
ACTION_SIZE = 2
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 3000000
BATCH_SIZE = 10 # every how many eposides do a parameter update
GAMMA = 0.99

# the abstraction of birdagent
class BirdAgent:
    def __init__(self, state_size, action_size, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.model = self._build_mode()

    def _build_mode(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                                input_shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)))  # 80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
        # get the 1 * 2 output represent each action's probability
        model.add(Activation('softmax'))

        return model

    # sample the action from the predict of the model
    def action_sampler(self, out):
        # return 1 if result > np.random.random() else 0
        return np.random.choice([0, 1], p=out)

    def act(self, state, epsilon = 0.01):
        explore = False
        random_num = random.random()
        action = np.zeros(2)
        if random_num <= epsilon:
            explore = True
            # ramdomly select an action
            action_index = random.randrange(ACTION_SIZE)
            action[action_index] = 1
            print("*********** Random Action *********** : ", action_index)
            return action,action,explore

        else:
            out = self.model.predict(state)
            if self.action_sampler(out=out) == 0:
                action[0] = 1
            else:
                action[1] = 1
            return np.squeeze(out),action,explore


# the abstraction of rollouts
class RollOuts():
    def __init__(self, next_states, actions, rewards, logprobs, gradients, total_steps):
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.steps = total_steps
        self.logprobs = logprobs
        self.gradients = gradients
        self.total_rewards = np.sum(rewards)


    def get_summary(self):
        return {" total_reward ": self.total_rewards,
                " total_steps ": self.steps}


    def prepare_target(self):
        result = []
        for action, adv in list(zip(self.actions, self.advs)):
            print('action and advantages : ',action, adv)
            result.append([action, adv])
        return np.array(result)


# the abstraction of experience memory
class ExperienceStroe:
    def __init__(self):
        self.rollouts = []

    def add_rollout(self, next_states, actions, rewards,logprobs, gradients, total_steps):
        self.rollouts.append(RollOuts(next_states, actions, rewards, logprobs, gradients,total_steps))

    def num_experiences(self):
        return len(self.rollouts)

    def get_range(self, start, end):
        return self.rollouts[start:end]

    def reset(self):
        self.rollouts = []


def stats_summary(rollouts, records, verbose=True):
    rollout_rewards = np.array([rollout.total_rewards for rollout in rollouts])
    print("reward mean %s" % (rollout_rewards.mean()))
    print("reward std %s" % (rollout_rewards.std()))
    print("reward max %s" % (rollout_rewards.max()))
    records.append([rollout_rewards.mean(), rollout_rewards.std()])


# calculate running reward
def cal_reward(rollouts, gamma, discounted=False):
    for rollout in rollouts:
        rewards = rollout.rewards
        r_reward = []
        running_reward = 0
        for reward in rewards[::-1]:
            # discounted reward
            reward = (-1) * reward
            if discounted == True:
                running_reward = gamma * running_reward + reward
            else:
                running_reward = running_reward + reward
            r_reward.append(running_reward)
        rollout.r_rewards = np.squeeze(np.array(np.vstack(r_reward[::-1])))


# calculate the advantage = reward - expected reward at this time step
def cal_advantage(rollouts):
    max_steps = max(rollout.rewards.shape[0] for rollout in rollouts)
    for rollout in rollouts:
        rollout.r_rewards = np.pad(rollout.r_rewards, (0, max_steps - rollout.r_rewards.shape[0]),'constant')
    baselines = np.mean(np.vstack([rollout.r_rewards for rollout in rollouts]), axis=0)
    for rollout in rollouts:
        rollout.advs = rollout.r_rewards - baselines
        rollout.advs = rollout.advs[:len(rollout.r_rewards)]
        rollout.r_rewards = rollout.r_rewards[:len(rollout.rewards)]


def play_game(agent, render=False):
    # 1. initialize
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # used to save the actions, should be a 1*2 nparray and sum(action) should be 1
    actions = np.array([])
    # used to save the rewards
    rewards = np.array([])
    # get the first state by doing nothing and preprocess the image to 80x80x4
    # In Keras, need to reshape 1 * 80 * 80 * 4
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])  # 1*80*80*4
    # the observations will be batch_size * 80 * 80 * 4
    observations = np.zeros([1, state.shape[1], state.shape[2], state.shape[3]])
    gradients = np.zeros([1, 2])
    # should be (1*2) nparray and each col represent the probability to chose that action
    logprobs = np.array([1, 2])


    do_nothing = np.zeros(ACTION_SIZE)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # preprocess the first frame
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    # rescale to 0-1
    x_t = x_t / 255.0

    # 80 * 80 * 4
    state = np.stack((x_t, x_t, x_t, x_t), axis=2)
    print (state.shape)


    for step in range(1, 500):
        # next state's shape, should be (1*80*80*4)
        print('observations size : ', observations.shape)
        print('state size ', state.shape)
        # get the state list
        observations = np.vstack((observations, state))
        # predict the action

        logprob,action,explore = agent.act(state)

        actions = np.append(actions, action)
        logprobs = np.vstack((logprobs ,logprob))
        gradients = np.vstack((gradients , action.astype('float32') - logprob))
        # use the predicted action to determine the next state
        x_t1_colored, reward, terminal = game_state.frame_step(action)
        # rgb to gray and rescale
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        # rescale to 0-1
        x_t1 = x_t1 / 255
        # update state
        x_t1 = x_t1.reshape(1,x_t1.shape[0],x_t1.shape[1],1)
        state = np.append(x_t1,state[:,:,:,:3],axis=3)
        rewards = np.append(rewards, reward)

        print('living steps : ',observations.shape[0])
        # print('state size ',state.shape)
        # print('action size : ',action.shape)
        # print('action : ',action)
        # print('reward : ',reward)
        # print('terminal : ',terminal)
        if terminal or step == 498:
            break
    return observations[1:], actions, rewards , logprobs[1:], gradients[1:], step


def play_n_games(agent, history, n=10, verbose=True):
    start_eps = history.num_experiences()
    total_step = 0
    for i in range(n):
        observations, actions, rewards, logprobs, gradients, step = play_game(agent=agent)
        history.add_rollout(observations, actions, rewards, logprobs, gradients, step)
        total_step += step
    end_eps = history.num_experiences()
    return start_eps, end_eps, total_step


