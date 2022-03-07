import numpy as np
import random
from collections import namedtuple

'''
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
style.use("ggplot")
'''


SIZE = 10



# Convenient data structure to hold information about actions
Action = namedtuple('Action', 'name index delta_i delta_j')
    
up = Action('up', 0, -1, 0)
down = Action('down', 1, 1, 0)
left = Action('left', 2, 0, -1)
right = Action('right', 3, 0, 1)

index_to_actions = {}
for action in [up, down, left, right]:
    index_to_actions[action.index] = action

str_to_actions = {}
for action in [up, down, left, right]:
    str_to_actions[action.name] = action



class SpaceShip:
    
    def __init__(self, N):    
        
        # Initialises 
        self.spaceship = np.zeros((N, N), dtype = np.int8)
        self.size = N
        self.end_game = False
        
        # Environment entitites key
        self.wall = 1 
        self.warppad = 2
        self.energycore = 3
        self.escapepod = 4
        self.agent = 5
        self.alien = 6      
        
        # Walls around placed around edge and randomlythroughout maze             
        self.spaceship[0, :] = self.wall
        self.spaceship[-1, :] = self.wall
        self.spaceship[:, 0] = self.wall
        self.spaceship[:, -1] = self.wall
        
        # Interior walls randomly placed throught ship
        empty_coords = self.get_empty_cells(int(self.size/2))
        self.spaceship[empty_coords[0], empty_coords[1]] = self.wall
        
        # Warpad locations
        empty_coords = self.get_empty_cells(1)
        self.warppadA = empty_coords
        self.spaceship[empty_coords[0], empty_coords[1]] = self.warppad
        
        empty_coords = self.get_empty_cells(1)
        self.warppadB = empty_coords
        self.spaceship[empty_coords[0], empty_coords[1]] = self.warppad
        
        # Energy cores are powerups so give good reward
        self.num_energycores = int(self.size/4)
        empty_coords = self.get_empty_cells(self.num_energycores)
        self.spaceship[empty_coords[0], empty_coords[1]] = self.energycore                
        
        # Current location and action for agent and alien
        self.agent_location = None
        self.alien_location = None
        
        self.agent_action = None
        self.alien_action = None
        
        # Available actions
        self.actions = {'up', 'down', 'left', 'right'}
   
        # Collsion used if agent hits one of the surrouding or interior walls 
        self.collision = False
        
        # Location of escape pod
        self.position_exit = self.get_empty_cells(1)
        self.spaceship[self.position_exit[0], self.position_exit[1]] = self.escapepod
        
        # Run time
        self.time_elapsed = 0
        self.time_limit = self.size**2
        
        # For spaceship display
        self.dict_map_display={ 0:'.',
                                1:'X',
                                2:'W',
                                3:'C',
                                4:'E',
                                5:'A',
                                6:'L'}


    # Step funtion performs a step in teh enviroment for the agent base on an action
    def step(self, action): 
        
        self.agent_action = action
        self.collision = False
        
        # Move agent
        new_location = self.move(self.agent_action, self.agent_location)
        
        # Agent remians at same location if it collides with a wall
        if self.coord_conversion(new_location) == 1:
            self.collision = True
        else:
            self.agent_location = new_location
        
        # Move alien
        self.alien_location = self.get_new_alien_location()
        
        # Give reward to agent
        reward = self.calculate_reward()
        
        # Calculate observations
        observations = self.calculate_observations()
        
        # Update time every step
        self.time_elapsed += 1
        
        # Set episode done to true if any end critiera met
        if self.time_elapsed == self.time_limit:
            print('GAMEOVER - Time Limit Reached')
            self.end_game = True
            
        if self.agent_location[0] == self.alien_location[0] and self.agent_location[1] == self.alien_location[1]:
            print('GAMEOVER - The Alien Killed You')
            self.end_game = True

        if (self.agent_location[0] == self.position_exit[0] and self.agent_location[1] == self.position_exit[1]) and self.num_energycores == 0:
            print('YOU ESCAPED!')
            self.end_game = True       
       
        return observations, reward, self.end_game
    
    # Moves the agent or alien based on the selected move
    def move(self, action, location):
        
        if action == 'up':
            new_location = np.array((location[0] - 1, location[1] ))
        elif action == 'down':
            new_location = np.array((location[0] + 1, location[1] ))
        elif action == 'left':
            new_location = np.array((location[0] , location[1] - 1 ))
        elif action == 'right':
            new_location = np.array((location[0] , location[1] + 1))   
        
        if new_location[0] == self.warppadA[0] and new_location[1] == self.warppadA[1]:
            new_location = self.warppadB
            
        if new_location[0] == self.warppadB[0] and new_location[1] == self.warppadB[1]:
            new_location = self.warppadA
        
        return new_location
    
    
    # Get aliens random action
    def get_new_alien_location(self):
        
        available_moves = []
        num1 = -1
        num2 = 0
        
        # Gets adjacent locations that the alien is available to move too
        for action in self.actions:
            location = self.move(action, self.alien_location)
            temp = self.coord_conversion(location)
            if temp != (1 or 3 or 4) :
                available_moves.append(location)
                num1 += 1
        
        # Randomly select one of the available moves   
        num2 = random.randint(0, num1)
        new_location = available_moves[num2]

        
        return new_location
    
    
    # Converts a pair of coordinates into the integer vlaue at that location
    def coord_conversion(self, location):
        
        value = self.spaceship[location[0], location[1]]
        
        return value
        
    
    # Calculate reward for agent for current move to new cell
    def calculate_reward(self):
        
        value = self.coord_conversion(self.agent_location)
        
        step_reward = 0
        
        # Reward for each time step is -1
        step_reward += -1
        
        # Reward for hitting a wall is -10
        if self.collision == True:
            step_reward += -10
        
        # Reward for being in the same location as the alien
        if self.agent_location[0] == self.alien_location[0] and self.agent_location[1] == self.alien_location[1]:
            step_reward += -100
        
        # Rewarad for collecting a energycore
        if value == 3:
            step_reward += 20
            self.spaceship[self.agent_location[0], self.agent_location[1]] = 0
            self.num_energycores -= 1
            
        # Reward for the agent winning
        if value == self.escapepod and self.num_energycores == 0:
            step_reward += 100
            
        return step_reward
        
        
    # Find cells in the environment that are currently empty
    def get_empty_cells(self, cells):
        
        empty_cells_coord = np.where(self.spaceship == 0)
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), cells)
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates
    
    
    # Calulates observations of agent
    def calculate_observations(self):
        
        relative_coordinates = self.position_exit - self.agent_location
                
        # Pad with zeros
        dungeon_padded = np.ones( (self.size + 2, self.size + 2), dtype = np.int8)
        dungeon_padded[1:self.size+1, 1:self.size+1] = self.spaceship[:,:]
        
        surroundings = dungeon_padded[ self.agent_location[0] + 1 -2: self.agent_location[0]+ 1 +3,
                                     self.agent_location[1]+ 1 -2: self.agent_location[1]+ 1 +3]
        
        
        obs = {'relative_coordinates':relative_coordinates,
               'surroundings': surroundings}
        
        return obs
    
    
    # Used to display current spaceship environment
    def display(self):
        
        envir_with_agent = self.spaceship.copy()
        envir_with_agent[self.agent_location[0], self.agent_location[1]] = self.agent
        envir_with_agent[self.alien_location[0], self.alien_location[1]] = self.alien
        
        full_repr = ""

        for r in range(self.size):
            
            line = ""
            
            for c in range(self.size):

                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                
                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"

        print(full_repr)
        
    
    # Resets spceship environment
    def reset(self):
       
        self.time_elapsed = 0
        
        # Position of the agent is a numpy array
        self.agent_location = np.asarray(self.get_empty_cells(1))
        self.alien_location = np.asarray(self.get_empty_cells(1))
        
        # Calculate observations
        observations = self.calculate_observations()
        
        return observations

# Runs a single game 
def run_single_exp(envir, policy):
    
    obs = envir.reset()
    done = False
    
    total_reward = 0
    
    while not done:
        
        action = policy(obs)
        obs, reward, done = envir.step(action)
        
        total_reward += reward
    
    return total_reward


       