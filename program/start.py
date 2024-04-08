import psutil
import threading
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt

from agent import DQNAgent
from gym_osu.envs.osu_env import OsuEnv
from memorywebsocket import MemoryWebSocket

# Initialize websocket for data
ws_endpoint = "ws://localhost:24050/ws"
ws = MemoryWebSocket(ws_endpoint)
threading.Thread(target=ws.run_ws).start()

# Initialize environment and agent
env = None
agent = None
episode = 0
episodes = 50
training_batch_size = 1 # Must be batch_size 1 due to only one snapshot per training step
epochs = 10
delta_epsilon = 0

def train_agent():
  train_step = 0
  while episode < episodes - 1:
    if len(agent.memory) >= training_batch_size:
      train_step += 1
      try:
        print(f'Training at step {train_step}')
        agent.train(epochs, training_batch_size)
      except Exception as e:
        print(f'Failed to train: Error: {e}')
        time.sleep(5)
      if train_step % 10 == 0:
        try:
          print(f'Updating target model at step {train_step}')
          agent.update_target()
        except Exception as e:
          print(f'Failed to update: Error: {e}')
          time.sleep(5)
      if train_step % 100 == 0:
        print(f'Saving checkpoint model at step {train_step}')
        agent.model.save_weights(f'./models/training_model_step{train_step}.h5')
  print('Training Stopped')
    
if __name__ == '__main__':
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

  env = OsuEnv(ws)
  agent = DQNAgent(env, lr=0.001, gamma=0.25)
  delta_epsilon = agent.epsilon/episodes
  
  time.sleep(1) # Give time to preprocess and load

  t1 = threading.Thread(target=train_agent).start()

  total_rewards = []
  epsilons = [agent.epsilon]

  step = 0
  for episode in range(episodes):
    print(f'Episode: {episode}')
    obs = env.reset()
    done = env.ws.complete
    epsilons.append(agent.epsilon)
    total_reward = 0
    last_target_update = 0

    while not done:
      action = agent.step(obs)
      state = obs.copy()
      obs, reward, done, info = env.step(action)
      total_reward += reward
      next_state = obs.copy()
      agent.collect(state, action, reward, done, next_state)
      step += 1
        
    agent.set_epsilon(agent.epsilon - delta_epsilon)
    total_rewards.append(total_reward)
  plt.title('Final Training Results')
  plt.xlabel('Episode')
  plt.ylabel('Total Reward')
  plt.plot([np.mean(total_rewards[tr]) for tr in range(len(total_rewards))])
  plt.show()

  print('Training complete, saving model...')
  t1.join()
  agent.model.save_weights('./models/finished_model.h5')
