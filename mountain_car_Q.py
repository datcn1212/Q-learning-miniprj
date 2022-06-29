from copyreg import pickle
from lib2to3.pytree import convert
import gym
import numpy as np
import pickle

env = gym.make("MountainCar-v0")
env.reset()

q_table_size = [20, 20] # trục tọa độ 2 chiều, mỗi chiêu lấy 20 điểm
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size

# chuyển đổi tử real-state về q-state
def convert_state(real_state):
    q_state = (real_state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(int))

# khởi tạo
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n])) #(20,20,3) -> 20x20 state, mỗi state 3 action
c_learning_rate = 0.1  # alpha
c_discount_value = 0.9 # gamma
c_no_of_eps = 10000    # number of episodes

v_epsilon = 0.9 # epsilon
c_start_ep_epsilon_decay = 1
c_end_ep_epsilon_decay = c_no_of_eps // 2
v_epsilon_decay = v_epsilon / (c_end_ep_epsilon_decay-c_start_ep_epsilon_decay)

max_ep_reward = -1000
max_ep_action_list = []
time = 1000

# train
for ep in range(c_no_of_eps):
    done = False
    current_state = convert_state(env.reset())
    ep_reward = 0
    action_list = []
    
    while not done:
        if np.random.random() > v_epsilon:
            # lấy max q-value của current_state
            action = np.argmax(q_table[current_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        action_list.append(action)
        
        # hành động theo action đã lấy
        next_real_state, reward, done, _ = env.step(action = action)
        ep_reward += reward
        
        if ep % time == 0:
            env.render() # render each (time) times -> 1000
        
        if done:
            if next_real_state[0] >= env.goal_position:
                print("Pass tại ep = {}, reward = {}".format(ep, ep_reward))
                if ep_reward > max_ep_reward:
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
                    # save q_table
                    # q_table_last = open("q_table_last.txt", "wb")
                    # pickle.dump(q_table, q_table_last)
            else:
                print("Fail tại ep = ", ep)
        else:
            # convert
            next_state = convert_state(next_real_state)
            
            # update Q value cho (current_state, action)
            current_q_value = q_table[current_state + (action,)]
            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(q_table[next_state]))
            q_table[current_state + (action,)] = new_q_value
            current_state = next_state
            
    if c_end_ep_epsilon_decay >= ep > c_start_ep_epsilon_decay:
        v_epsilon -= v_epsilon_decay   
         
print("max reward = ", max_ep_reward)
print("max action list = ", max_ep_action_list)