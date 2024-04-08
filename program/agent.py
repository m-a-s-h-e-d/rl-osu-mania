import collections
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class MLP(keras.Model):
  def __init__(self, observation_size, action_size):
    super(MLP, self).__init__()
    self.observation_size = observation_size
    self.action_size = action_size
    
    self.value_dense_1 = layers.Dense(128, activation='relu')
    self.value_dense_2 = layers.Dense(128, activation='relu')
    self.values = layers.Dense(self.action_size)

  def call(self, inputs):
    input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
    y = self.value_dense_1(input_tensor)
    y = self.value_dense_2(y)
    values = self.values(y)
    return values

class DQNAgent:
  def __init__(self, env, epsilon=1.0, epsilon_min=0.02, lr=5e-4, gamma=0.99, memory_limit=1024):
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.lr = lr
    self.gamma = gamma
    self.memory = collections.deque([], memory_limit)
    self.memory_limit = memory_limit
    
    try:
      model_input_dim = self.observation_space.n
      model_input_shape = (self.observation_space.n, 1)
    except AttributeError as e:
      model_input_dim = self.observation_space.shape[0]
      model_input_shape = self.observation_space.shape
    
    self.model = MLP(model_input_dim, self.action_space.n)
    self.model(tf.convert_to_tensor([np.random.normal(size=model_input_shape)], dtype=tf.float32))
    
    self.target_model = MLP(model_input_dim, self.action_space.n)
    self.target_model(tf.convert_to_tensor([np.random.normal(size=model_input_shape)], dtype=tf.float32))
      
  def step(self, observation, verbose=False):
    if np.random.uniform(0, 1) < self.epsilon:
      return np.random.choice(self.action_space.n)
    else:
      state_tensor = tf.convert_to_tensor([observation], dtype=tf.float32)
      q_vals = self.model(state_tensor)
      if verbose:
        print(q_vals[0])
      return np.argmax(q_vals[0])
      
  def collect(self, state, action, reward, done, next_state):
    self.memory.append((state, action, reward, done, next_state))
      
  def sample(self, size):
    sample_index = np.random.choice(len(self.memory), size=size, replace=True)
    
    states = []
    actions = []
    rewards = []
    dones = []
    next_states = []
    for idx in sample_index:
      unit = self.memory[idx]
      states.append(unit[0])
      actions.append(unit[1])
      rewards.append(unit[2])
      dones.append(unit[3])
      next_states.append(unit[4])
    return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)
      
  def train(self, epochs=10, batch_size=128, verbose=False):
    losses = []
    for epoch in range(epochs):
      states, actions, rewards, dones, next_states = self.sample(batch_size)
      states = tf.convert_to_tensor(states, dtype=tf.float32)
      done_masks = tf.convert_to_tensor((~dones.astype(bool)).astype(int), dtype=tf.float32)
      next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
      action_masks = tf.one_hot(actions, self.action_space.n, dtype=tf.float32)
      target_values = tf.expand_dims( rewards + self.gamma * np.max( self.target_model( next_states ) ) * done_masks, axis = 1 )
      target_values *= action_masks
      with tf.GradientTape() as tape:
        q_values = self.model(states) * action_masks
        if verbose:
          print(target_values, q_values)
        loss = tf.reduce_mean((target_values - q_values)**2)

      losses.append(loss)
      grads = tape.gradient(loss, self.model.trainable_variables)
      optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
      optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return np.mean(losses)
          
  def update_target(self):
    self.target_model.set_weights(self.model.get_weights())
                  
  def set_epsilon(self, epsilon):
    self.epsilon = epsilon
    if self.epsilon < self.epsilon_min:
      self.epsilon = self.epsilon_min
