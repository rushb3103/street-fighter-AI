import retro,pygame,retrowrapper
from gym import Env
from gym.spaces import Discrete, Box, MultiBinary
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import Wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED,players = 2)
        #self.score = 0
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Preprocess frame from game
        frame_delta = obs 
#         - self.previous_frame
#         self.previous_frame = obs 
        
        # Shape reward
        # reward = info['score'] - self.score 
        # self.score = info['score']
        reward = (self.enemy_health - info['enemy_health'])*2 + (info['health'] - self.health)

        return frame_delta, reward, done, info 
    
    def render(self, *args, **kwargs): 
        self.game.render(*args, **kwargs)
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.health = 176
        self.enemy_health = 176
        
        # Create initial variables
        self.score = 0

        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84,84,1))
        return state
    
    def close(self): 
        self.game.close()
    

model = PPO.load('D:/Project/train/best_model_334000.zip')
#model = PPO.load('./train/best_model_170000.zip')


env = StreetFighter()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

#pygame
pygame.init()
display_width = 200
display_height = 200
win = pygame.display.set_mode((display_width,display_height))

# j = pygame.joystick.Joystick(0)
# j.init()

clock = pygame.time.Clock()

butts = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

action_array = [0,0,0,1,0,0,0,0,0,0,0,0]

obs = env.reset()
total_reward = 0
done = False

def create_text(action_list):
    global display_height,display_width,win,butts
    on_color = (0,255,0)
    off_color = (255,0,0)
    inputlist1 = butts[4:8]
    inputlist2 = butts[8:]
    marginx = 30
    marginy = 10
    text_height = (display_height-marginy)/4
    text_width = (display_width-marginx)/2

    for i in range(2):
        left = 0 + (100*i) + marginx
        for j in range(4):
            right = 0 + (50*j) + marginy
            font = pygame.font.SysFont("arial", 20)
            str = ''
            if i > 0:
                str = inputlist2[j]
            else:
                str = inputlist1[j]
            # print(left,right)
            text = font.render(str,True,off_color)
            if str in action_list:
                text = font.render(str,True,on_color)
            win.blit(text,(left,right))



def create_box():
    global display_height,display_width,win
    margin = 5
    box_height = (display_height-margin)/4
    box_width = (display_width-margin)/2

    for i in range(2):
        left = 0 + (100*i) + margin
        for j in range(4):
            right = 0 + (50*j) + margin
            rect = pygame.Rect((left,right),(box_width,box_height))
            pygame.draw.rect(win,(255,255,255),rect)


while not done:
    
    actions = set()
   
    # Control Events
    
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    for event in pygame.event.get():
        
        if keys[pygame.K_RIGHT]:
            actions.add('RIGHT')
        if keys[pygame.K_LEFT] :
            actions.add('LEFT') 
        if keys[pygame.K_DOWN] :
            actions.add('DOWN')
        if keys[pygame.K_UP]:
            actions.add('UP')            
      
        if keys[pygame.K_z] :
            actions.add('A')
        if keys[pygame.K_x] :
            actions.add('B')
        if keys[pygame.K_c] :
            actions.add('C')
        if keys[pygame.K_a] :
            actions.add('X')
        if keys[pygame.K_s] :
            actions.add('Y')
        if keys[pygame.K_d] :
            actions.add('Z')
        if keys[pygame.K_SPACE]:
            actions.add('START')

    
        for i, a in enumerate(butts):
            if a in actions:
                action_array[i] =  1            
            else:
                action_array[i] = 0
            
 
    # time.sleep(2)
    a2, _ = model.predict(obs)
    # print(type(a2),a2)
    # env.render()
    
    action_array = np.array(action_array)

    
    #Player input
    # act = np.concatenate(([action_array],a2), axis=1)

    # AI input
    act = np.concatenate((a2,[action_array]), axis=1)

    action_list = []
    action = ''
    # print(act[0,0:6])
    ar1 = act[0,0:12]
    # print(action_array)
    for index,value in enumerate(ar1):
        if value == 1:
            action = butts[index]
            action_list.append(action)
            # print(action_list)
            # print(butts[index])
    
    # print(action)
    create_box()
    create_text(action_list)

    
    # print(type(act),act)
    # Progress Environemt forward
    obs, rew, done, info = env.step(act)
    total_reward += rew

    env.render()
  
    clock.tick(60)
    pygame.display.update()

        

env.close()
pygame.QUIT