import random
import psutil
import threading
import yaml

from gym_osu.envs.osu_env import OsuEnv
from memorywebsocket import MemoryWebSocket

# Read the YAML file
with open('./settings.yml', 'r') as file:
  data = yaml.safe_load(file)
if not data:
  raise Exception(f'Data missing from settings.yml, redownload and try again')

# Check that osu! and gosumemory are running processes
running_processes = (process.name() for process in psutil.process_iter())
for process in data['processes'].values():
  if not process in running_processes:
    raise ProcessLookupError(f'Process {process} is not running, please run it and try again.\n'\
                             f'If this issue persists, check the process names in settings.yml.')
  
ws_endpoint = data['endpoint']['gosumemory']

# Setup environment
def start_game(ws):
  env = OsuEnv(ws)
  epsilon = 0.1
  decay = 0.99998
  steps = 10000

  # Start a track here
  # Wait until track has started before starting iterations
  while not env.ws.playing:
    pass

  for _ in range(0, steps):
    if random.random() < epsilon:
        env.step(env.action_space.sample())
        epsilon *= decay
    else:
      try:
        env.step(env.model.get_qs(env.previous_observation))
      except:
        pass

    if env.ws.complete:
      env.reset()
      env.model.save('models/current.h5')

def start_program():
  ws = MemoryWebSocket(ws_endpoint)
  threading.Thread(target=ws.run_ws).start()
  start_game(ws)

if __name__ == '__main__':
  start_program()