#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""


import os
import numpy as np
import random as rn
import environment
from keras.models import load_model

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0),
                              initial_month = 0,
                              initial_number_users = 20,
                              initial_rate_data = 30)

# LOADING TRAINED BRAIN
model = load_model("model.h5")

# CHOOSING THE MODE
train = False

# RUNNING A 1 YEAR SIMULATION
env.train = train
current_state, _, _ = env.observe()
initial_month = np.random.randint(0, 12)
for timestep in range(0, 1 * 30 * 24 * 60):
    q_values = model.predict(current_state)[0]
    action = np.argmax(q_values)
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    # energy_ai = abs(action - direction_boundary) * temperature_step
    energy_ai = 0
    month = initial_month + int(timestep / (30 * 24 * 60))
    if month >= 12:
        month -= 12
    next_state, reward, game_over = env.update_env(direction, energy_ai, month)
    current_state = next_state
    
print("\n")
print("Total energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total energy spent with no AI: {:.0f}".format(env.total_energy_noai))

print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
