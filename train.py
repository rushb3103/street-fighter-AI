import retro,time
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Importing the optimzation frame - HPO
import optuna
# PPO algo for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging 
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to vectorize and frame stack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# Import os to deal with filepaths
import os
# Import base callback 
from stable_baselines3.common.callbacks import BaseCallback


LOG_DIR = './logs/'
OPT_DIR = './opt/'
CHECKPOINT_DIR = './train/'

#custom environment

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

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# Create environment 
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model_params = {'n_steps': 4092,
  'gamma': 0.8469613057339491,
  'learning_rate': 1.688874996274341e-05,
  'clip_range': 0.186254787282314,
  'gae_lambda': 0.8516711838664867}
model_params['n_steps'] = 4032  # set n_steps to 7488 or a factor of 64
# model_params['learning_rate'] = 5e-7

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)

model.load(os.path.join(OPT_DIR, 'trial_0_best_model.zip'))

# Kick off training 
model.learn(total_timesteps=5000000, callback=callback)