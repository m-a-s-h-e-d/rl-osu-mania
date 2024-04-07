import cv2
import gym
import numpy as np
import pyautogui
import pydirectinput
import threading
import yaml

from PIL import ImageGrab
from agent import DQNAgent

# Read the YAML file
with open('./settings.yml', 'r') as file:
  data = yaml.safe_load(file)
if not data:
  raise Exception(f'Data missing from settings.yml, redownload and try again')

keys = [None, None, None, None]
keys[0] = data['keybinding']['key1']
keys[1] = data['keybinding']['key2']
keys[2] = data['keybinding']['key3']
keys[3] = data['keybinding']['key4']

# Manually adjust this to your own screen size and crop to the playfield
screen_width, screen_height = pyautogui.size()
ratio = float(2.8)
left_padding = int(screen_width / ratio)
right_padding = screen_width - left_padding
bottom_height = screen_height - int(screen_height / 2)
window_width = right_padding - left_padding
window_height = screen_height - bottom_height
width = int(window_width * 0.07)
height = int(window_height * 0.07)

key_combos = [[]]
# Combinations for all key presses (!len(keys)), technically constant time
for i in range(0, len(keys)):
  key_combos.append((keys[i]))
  for j in range(i + 1, len(keys)):
    key_combos.append((keys[i], keys[j]))
    for k in range(j + 1, len(keys)):
      key_combos.append((keys[i], keys[j], keys[k]))
      for l in range(k + 1, len(keys)):
        key_combos.append((keys[i], keys[j], keys[k], keys[l]))

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

class OsuEnv(gym.Env):
  def __init__(self, ws):
    super().__init__()
    self.ws = ws
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(bottom_height, right_padding - left_padding), dtype=np.uint8)
    self.action_space = gym.spaces.Discrete(len(key_combos))
    self.model = DQNAgent(self)
    self.previous_observation = None

  def step(self, action):
    def press_key(key):
      pydirectinput.press(key)

    threads = []
    
    for key in key_combos[action]:
      t = threading.Thread(target=press_key, args=(key,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()
    
    reward = self.ws.get_reward()
    print(f'Action: {action} : Reward: {reward}')

    
    im = ImageGrab.grab(bbox =(left_padding, bottom_height, right_padding, screen_height)) 
    shot = np.array(im)
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    gray = resize_frame(gray, width, height)
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
    self.previous_observation = None
    return np.zeros((screen_height, right_padding - left_padding))
