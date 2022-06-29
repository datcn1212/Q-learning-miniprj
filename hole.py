import numpy as np
import random
import matplotlib.pyplot as plt

# rows and columns length
BOARD_ROWS = 5
BOARD_COLS = 5

# start, win and lose states
START = (0,0)
WIN_STATE = (4,4)
HOLE_STATE = [(1,0),(3,1),(4,2),(1,3)]

# environment
class State:
    def __init__(self, state=START):
        #initalise the state to start and end to false
        self.state = state
        self.isEnd = False        

    def getReward(self):
        #give the cost for each state -5 for loss, +1 for win, -1 for others
        for i in HOLE_STATE:
            if self.state == i:
                return -5
        if self.state == WIN_STATE:
            return 1           
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def nxtPosition(self, action):     
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1]) #up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1]) #down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1) #left
        else:
            nxtState = (self.state[0], self.state[1] + 1) #right

        #check if next state is possible
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):                  
                    return nxtState      
        return self.state 

         
class Agent:

    def __init__(self):
        #inialise states and actions 
        self.states = []
        self.actions = [0,1,2,3]    # up, down, left, right
        self.State = State()
        
        #set the learning and greedy values
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd

        # array to retain reward values for plot
        self.plot_reward = []
        
        #initalise Q values as a dictionary for current and new
        self.Q = {}
        # initalise cost to 0
        self.cost = 0
        
        # initalise all Q values across the board to 0, print these values
        self.Q = np.zeros([BOARD_ROWS, BOARD_COLS, len(self.actions)])
    

    def Action(self):
        # random value vs epsilon
        rnd = random.random()
        action = None
        
        # choose max-Q-value action 
        if(rnd > self.epsilon) :
            action = np.argmax(self.Q[self.State.state])
                    
        # random action
        else:
            action = np.random.choice(self.actions)
        
        # action to next state
        position = self.State.nxtPosition(action)
        return position,action
    
    
    # Q-learning Algorithm
    def Q_Learning(self, episodes):
        
        costt = -999
        b = []
        for ep in range(episodes):
            #reset
            self.State = State()
            self.isEnd = self.State.isEnd 
            self.cost = 0
            a= []
            
            while not self.isEnd:
                next_state, action = self.Action()
                a.append(action)
                i,j = self.State.state
                reward = self.State.getReward()
                self.cost += reward
                
                new_q_value = (1-self.alpha)*self.Q[(i,j,action)] + self.alpha*(reward + self.gamma*np.max(self.Q[next_state]))
                # update Q values 
                self.Q[(i,j,action)] = round(new_q_value,3)
                
                # next state is now current state, check if end state
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                             
            # get current rewrard and add to array for plot
            reward = self.State.getReward()
            self.cost += reward
            self.plot_reward.append(self.cost)
            if self.cost > costt:
                costt = self.cost
                b = a
          
            print("ep = {}, cost = {}, a = {}".format(ep,self.cost,a))
                        
        #print final Q table output
        print(self.Q)
        print(costt, b)
        
    #plot the reward vs episodes
    def plot(self):
        plt.plot(self.plot_reward)
        plt.show()
            
    # largest Q value in each, print output
    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                mx_nxt_value = np.max(self.Q[i,j])
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')
        
        
if __name__ == "__main__":
    ag = Agent()
    episodes = 10000
    ag.Q_Learning(episodes)
    ag.plot()
    ag.showValues()