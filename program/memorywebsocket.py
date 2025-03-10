import json
import websocket as websockets

from enums.state import GameMode, GameState
from pynput.keyboard import Controller

GEKI_SCORE = 300
THREE_HUNDRED_SCORE = 200
SLIDER_BREAK_SCORE = 100
ONE_HUNDRED_SCORE = 25
FIFTY_SCORE = 10
ZERO_SCORE = 0 # -100, temporarily removing negative score to prevent it from not pressing

class MemoryWebSocket():
  def __init__(self, ws_endpoint):
    self.ws_endpoint = ws_endpoint
    self.websocket = None
    self.thread_ws_stop = False
    self.keyboard = Controller()
    self.env = None
    self.data = None
    self.state = None
    self.playing = False
    self.complete = False
    self.prev_hp = 0
    self.hp = 0
    self.prev_hits = [0, 0, 0, 0, 0, 0]
    self.hits = [0, 0, 0, 0, 0, 0] # PERFECT, 300, sliderBreak (200), 100, 50, 0

  def reset_state(self):
    print("Resetting state")
    self.complete = False
    self.playing = False
    self.hits = [0, 0, 0, 0, 0, 0]
    self.prev_hits = [0, 0, 0, 0, 0, 0]

  def check_state(self):
    if self.state == GameState.PLAYING.value:
      game_mode = self.data['gameplay']['gameMode']
      # Check that the game mode is correct
      if game_mode is not GameMode.MANIA.value:
        print('Game mode is not implemented')
        return
      self.playing = True
      self.complete = False
      self.hits[0] = self.data['gameplay']['hits']['geki']
      self.hits[1] = self.data['gameplay']['hits']['300']
      self.hits[2] = self.data['gameplay']['hits']['sliderBreaks']
      self.hits[3] = self.data['gameplay']['hits']['100']
      self.hits[4] = self.data['gameplay']['hits']['50']
      self.hits[5] = self.data['gameplay']['hits']['0']
      self.prev_hp = self.hp
    else:
      # Check for completion
      if self.playing:
        self.playing = False
        self.complete = True

  def get_reward(self):
    reward = 0
    # print(f'Previous: {self.prev_hits}')
    # print(f'Now: {self.hits}')
    if self.hits[0] > self.prev_hits[0]:
      reward += GEKI_SCORE
    if self.hits[1] > self.prev_hits[1]:
      reward += THREE_HUNDRED_SCORE
    if self.hits[2] > self.prev_hits[2]:
      reward += SLIDER_BREAK_SCORE
    if self.hits[3] > self.prev_hits[3]:
      reward += ONE_HUNDRED_SCORE
    if self.hits[4] > self.prev_hits[4]:
      reward += FIFTY_SCORE
    if self.hits[5] > self.prev_hits[5]:
      reward += ZERO_SCORE
    self.prev_hits = self.hits.copy()
    return reward
 
  def on_message(self, ws, message):
    if self.thread_ws_stop:
      ws.close()
    
    try:
      # Convert response to a Python object
      self.data = json.loads(message)
      # Check state and then perform a callback to send to model
      self.state = self.data['menu']['state']
      self.check_state()
    except Exception as e:
      print(e)

  # Hook websocket connection to gosumemory (ws://localhost:24050/ws)
  def run_ws(self):
    print(f'Listening to websocket endpoint: {self.ws_endpoint}')
    websocket = websockets.WebSocketApp(self.ws_endpoint,
                      on_message=self.on_message)
    websocket.run_forever()
