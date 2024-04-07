import gym
import cv2
import pyautogui
import numpy as np
import yaml

from PIL import ImageGrab
from agent import DQNAgent

# Read the YAML file
with open('./settings.yml', 'r') as file:
  data = yaml.safe_load(file)
if not data:
  raise Exception(f'Data missing from settings.yml, redownload and try again')

key1 = data['keybinding']['key1']
key2 = data['keybinding']['key2']
key3 = data['keybinding']['key3']
key4 = data['keybinding']['key4']
KEYS = 4

# Manually adjust this to your own screen size and crop to the playfield
screen_width, screen_height = pyautogui.size()
ratio = float(2.8)
left_padding = int(screen_width / ratio)
right_padding = screen_width - left_padding
bottom_height = screen_height - int(screen_height / 2)

class OsuEnv(gym.Env):
  def __init__(self, ws):
    super().__init__()
    self.ws = ws
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(bottom_height, right_padding - left_padding), dtype=np.uint8)
    self.action_space = gym.spaces.Discrete(KEYS + 1)
    self.model = DQNAgent(self)
    self.previous_observation = None

  def step(self, action):
    if action == 0:
      pyautogui.keyDown(key1)
      pyautogui.keyUp(key1)
      reward = self.ws.get_reward()
    elif action == 1:
      pyautogui.keyDown(key2)
      pyautogui.keyUp(key2)
      reward = self.ws.get_reward()
    elif action == 2:
      pyautogui.keyDown(key3)
      pyautogui.keyUp(key3)
      reward = self.ws.get_reward()
    elif action == 3:
      pyautogui.keyDown(key4)
      pyautogui.keyUp(key4)
      reward = self.ws.get_reward()
    elif action == 4:
      # No key press
      reward = self.ws.get_reward()
    print(f'Action: {action} : Reward: {reward}')

    shot = np.array(ImageGrab.grab(bbox =(left_padding, 0, right_padding, screen_height)))
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    if self.previous_observation is not None:
        self.model.update_replay_memory((self.previous_observation, action, reward, gray))
    
    self.previous_observation = gray

    return self.previous_observation, reward, self.ws.complete, []
  
  def reset(self):
    print("Completed")
    self.ws.reset_state()
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    pyautogui.sleep(1)
    pyautogui.keyDown('enter')
    pyautogui.keyUp('enter')
    self.ep_return = 0
    self.previous_observation = None
    return np.zeros((screen_height, right_padding - left_padding))
