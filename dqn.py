#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""


import numpy as np

# IMPLEMENTING DEEP Q-LEARNING WITH EXPERIENCE REPLAY

class DQN(object):
    
    # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE DQN
    def __init__(self, max_memory = 100, discount = 0.9):
        
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        
        
    # MAKING A METHOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY
    def remember(self, transition, game_over):
        
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
            
    
    # MAKING A METHOD THAT BUILDS TWO BATCHES OF INPUTS AND TARGETS BY YEXTRACTING TRANSITIONS FROM THE MEMORY
    def get_batch(self, model, batch_size = 10):
        
        len_memory = len(self.memory)
        # premier objet dans la memoire ,transition  et premier objet de la transition
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        
        transitions = np.random.randint(0, len_memory, size = min(len_memory, batch_size))
        
        for i, idx in enumerate(transitions):
            
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]   
            
            if game_over == 1:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * np.max(model.predict(next_state)[0])
                
        return inputs, targets
