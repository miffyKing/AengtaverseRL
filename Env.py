"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class EcoSystemEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8                 | 4.8                |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |
    **Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is 475 for v1.
    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ### Episode Termination
    The episode terminates if any one of the following occurs:
    1. Pole Angle is greater than ±12°
    2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Episode length is greater than 500 (200 for v0)
    ### Arguments
    ```
    gym.make('CartPole-v1')
    ```
    No additional arguments are currently supported.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        high = np.array(
            [
                5, 5, 5, 10, 10,
                10, 10, 10, 15, 15,
            ],
            dtype=np.int,
        )

        self.action_space = spaces.Discrete(20)  # 10 species inc/dec 0~19
        self.observation_space = spaces.Box(0, high, dtype=np.int)
        self.state = None
        self.tick = 0

    def simulate(self):
        self.tick = np.random.randint(5)
        return self.tick

    def step(self, action):
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = self.state
        if action == 0:
            a0 += 1
        elif action == 1:
            a1 += 1
        elif action == 2:
            a2 += 1
        elif action == 3:
            a3 += 1
        elif action == 4:
            a4 += 1
        elif action == 5:
            a5 += 1
        elif action == 6:
            a6 += 1
        elif action == 7:
            a7 += 1
        elif action == 8:
            a8 += 1
        elif action == 9:
            a9 += 1
        elif action == 10:
            a0 -= 1
        elif action == 11:
            a1 -= 1
        elif action == 12:
            a2 -= 1
        elif action == 13:
            a3 -= 1
        elif action == 14:
            a4 -= 1
        elif action == 15:
            a5 -= 1
        elif action == 16:
            a6 -= 1
        elif action == 17:
            a7 -= 1
        elif action == 18:
            a8 -= 1
        elif action == 19:
            a9 -= 1

        self.state = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
        print(self.state)
        sim_tick = self.simulate()
        done = bool(
            (sim_tick > 5) or
            not (a0 and a1 and a2
             and a3 and a4
             and a5 and a6
             and a7 and a8 and a9) # Max 값 넘어가는 것도 설정해야함
        )
        print("sim_tick",sim_tick)
        if not done:
            reward = 1
        elif sim_tick > 5:
            reward = 10
        else:
            reward = 0
        return np.array(self.state, dtype=np.int), reward, done

    def reset(self, animal_array):
        self.state = animal_array
