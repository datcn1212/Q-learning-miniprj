import gym

# tạo biến môi trường
env = gym.make("MountainCar-v0")
env.reset()

# lấy state hiện tại sau khởi tạo
print(env.state)
#env.render() #show môi trường
#input()

# lấy số action mà xe có thể thực hiện
print(env.action_space.n)

# lấy biên độ tối thiểu, tối đa và vận tốc tối thiểu, tối đa
print(env.observation_space.high, env.observation_space.low)

# render thử
while True:
    action = 2 # luôn đi về phải
    new_state, reward, done, _ = env.step(action)
    print("new state = {}, reward = {}, done = {}".format(new_state, reward, done))
    env.render()