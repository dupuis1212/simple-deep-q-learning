#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""

# Importing the libraries and the other python files
import os
import numpy as np
import random as rn
import environment
import brain
import dqn

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


# SETTING THE PARAMETERS
epsilon = 0.3   
number_actions = 5
# indice du mileu pour calculer la direction
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5


# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0),
                              initial_month = 0,
                              initial_number_users = 20,
                              initial_rate_data = 30)

# BUILDING THE BRAIN BY SIMPLY CREATING AN OBJECT OF THE BRAIN CLASS
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# BUILDING THE DQN MODEL BY SIMPLY CREATING AN OBJECT OF THE DQN CLASS
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

# CHOOSING THE MODE
train = True


# TRAINING THE AI
env.train = train
model = brain.model

if (env.train):
    
    # STARTING THE LOOP OVER ALL THE EPOCHS (1 Epoch = 5 Months)
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        
        while (timestep <= 5 * 30 * 24 * 60 and (not game_over)):
            
            # PLAYING THE NEXT ACTION BY EXPLORATION
            
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action < direction_boundary):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            
            # PLAYING THE NEXT ACTION BY INFERENCE
            else:
                q_values = model.predict(current_state)[0]
                action = np.argmax(q_values)
                if (action < direction_boundary):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step 

            
            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            month = new_month + int(timestep / (30 * 24 * 60))
            if month >= 12:
                month -= 12
                
            next_state, reward, game_over = env.update_env(direction, energy_ai, month)
            
            total_reward += reward
            
            # STORING THIS NEW TRANSITION INTO THE MEMORY
            transition = [current_state, action, reward, next_state]
            dqn.remember(transition, game_over)      
            
            # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            
            # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
        # PRINTING THE TRAINING RESULTS FOR EACH EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        # SAVING THE MODEL
        model.save("model.h5")





























