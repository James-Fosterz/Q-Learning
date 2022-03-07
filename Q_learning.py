import numpy as np
import random

from spaceship import SpaceShip

# Q learnign class
class Q_Learning():
    def __init__(self, EPOCHS, ENV):
        # Env is N*N for all agent locations then N*N for all alien locations then fianlly the 4 possible moves for the agent
        # number of q values equal N to the power 4 then times by 4
        self.env = ENV
        self.q_table = np.zeros([self.env.size*self.env.size, self.env.size*self.env.size, 4])
        self.alpha = 0.99
        self.gamma = 0.99
        self.epochs = EPOCHS
        self.epsilon = 1 
        self.action_dict = ["up", "down","left", "right"]
        self.lr = 0.1
        self.rewards = []
    
    # Runs an environment once until it has finished updating the q table as it goes
    def main(self):
        
        # Resets the env so that for each epoch the same env is use from the start
        self.env = ENV
        
        # Runs games in the env till completion
        # Each game is 1 epoch run till completion
        for epoch in range(self.epochs):
            
            # State is set to the agents locations and liekwise for the alien
            state = self.env.agent_location
            alien_state = self.env.alien_location
            
            # These state coordinates is then converted to a numerical value corresponding to a row in the q table
            row = (state[0]*self.env.size + state[1])
            alien_row = (alien_state[0]*self.env.size + alien_state[1]) 
            
            epoch_reward = 0
            
            # Runs the game until one of the en game critieria is met
            while not self.env.end_game:
                
                row = (state[0]*self.env.size + state[1]) 
                alien_row = (alien_state[0]*self.env.size + alien_state[1]) 
                
                action = self.e_greedy(state)
                _, reward, _ = self.env.step(action)
                new_state = self.env.agent_location
                new_row = (new_state[0]*self.env.size + new_state[1])
                index_action = self.action_dict.index(action)
                
                q_old = self.q_table[row, alien_row, index_action]
                q_next = np.max(self.q_table[new_row, alien_row])
                
                # Updates q value for in q table for the move
                self.q_table[row, alien_row, index_action] = q_old * (1 - self.lr) + self.lr * (reward + self.gamma * q_next)             
                
                # Changes the current state to the next state
                state = new_state
                
                # Adds the step reward to the epoch reward
                epoch_reward += reward  
                
            # Resets the end criteria and then displays dungeon
            self.env.end_game = False
            self.rewards.append(epoch_reward)
            #self.env.display()
            
        return self.rewards

                
                
    # E greedy policy used for taking actions
    def e_greedy(self, state):
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            index_action = np.argmax(self.q_table[state])
        else:
            index_action = random.randint(0, 3)
        
        action = self.action_dict[index_action]
        return action


        
N = 10

ENV = SpaceShip(N)
ENV.reset()


EPOCHS = 1000

Q = Q_Learning(EPOCHS, ENV)
rewards = Q.main()

#print(rewards)