"""
Solving USV guidance using actor-critic model
"""

import gym
import gym_usv.envs
import numpy as np
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Input
from tensorflow.compat.v1.keras.layers import Add, Concatenate, BatchNormalization
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.initializers import RandomUniform
import tensorflow.compat.v1.keras.backend as K

import tensorflow.compat.v1 as tf

import random
from collections import deque

tf.disable_eager_execution()

def stack_samples(samples):
	array = np.array(samples)
	
	current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
	actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
	rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
	new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
	
	return current_states, actions, rewards, new_states, dones
	

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .99
		self.tau   = .001

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=1000000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32,
			[None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output,
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output,
			self.critic_action_input) # where we calcaulte de/dC for feeding above

		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		#state_input_b = BatchNormalization()(state_input)
		h1 = Dense(400, activation='relu', bias_initializer='glorot_uniform')(state_input)
		#h1 = BatchNormalization()(h1)
		h2 = Dense(300, activation='relu', bias_initializer='glorot_uniform')(h1)
		#h2 = BatchNormalization()(h2)
		output = Dense(self.env.action_space.shape[0], activation='tanh', kernel_initializer=RandomUniform(-3e-3, 3e-3), bias_initializer=RandomUniform(-3e-3, 3e-3))(h2)

		model = Model(inputs=state_input, outputs=output)
		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		#state_input_b = BatchNormalization()(state_input)
		state_h1 = Dense(400, activation='relu', bias_initializer='glorot_uniform')(state_input)
		#state_h1 = BatchNormalization()(state_h1)
		state_h2 = Dense(300, activation='relu', bias_initializer='glorot_uniform')(state_h1)
		#state_h2 = BatchNormalization()(state_h2)

		action_input = Input(shape=self.env.action_space.shape)
		#action_input_b = BatchNormalization()(action_input)
		action_h1    = Dense(300, activation='relu', bias_initializer='glorot_uniform')(action_input)
		#action_h1    = BatchNormalization()(action_h1)

		merged = Add()([state_h2, action_h1])
		#merged = BatchNormalization()(merged)
		merged_h2 = Dense(300, activation='relu', bias_initializer='glorot_uniform')(merged)
		output = Dense(1, activation='linear', kernel_initializer=RandomUniform(-3e-3, 3e-3), bias_initializer=RandomUniform(-3e-3, 3e-3))(merged_h2)
		model  = Model(inputs=[state_input,action_input], outputs=output)

		adam  = Adam(lr=0.001, decay=0.01)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		
			cur_states, actions, rewards, new_states, _ =  stack_samples(samples)
			predicted_actions = self.actor_model.predict(cur_states)*np.pi/2
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_states,
				self.critic_action_input: predicted_actions
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_states,
				self.actor_critic_grad: grads
			})

	def _train_critic(self, samples):
   

		cur_states, actions, rewards, new_states, dones = stack_samples(samples)
		target_actions = self.target_actor_model.predict(new_states)*np.pi/2
		future_rewards = self.target_critic_model.predict([new_states, target_actions])
		
		rewards += self.gamma * future_rewards * (1 - dones)
		
		self.evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
		#print(evaluation.history)
	def train(self):
		batch_size = 64
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self.samples = samples
		self._train_critic(samples)
		self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
		self.target_critic_model.set_weights(critic_target_weights)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		'''self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()'''
		return self.actor_model.predict(cur_state)*np.pi/2


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


sess = tf.Session()
K.set_session(sess)
env = gym.make("usv-asmc-v0")
actor_critic = ActorCritic(env, sess)
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))

num_trials = 10000
trial_len  = 400

starting_weights = 0
if starting_weights == 0:
	print("Starting on new weights")
else:
	actor_critic.actor_model.load_weights("./ddpg_models/iteration" + str(starting_weights))
	actor_critic.critic_model.load_weights("./ddpg_models/critic/critic" + str(starting_weights))
	actor_critic.target_actor_model.load_weights("./ddpg_models/target_actor/target_actor" + str(starting_weights))
	actor_critic.target_critic_model.load_weights("./ddpg_models/target_critic/target_critic" + str(starting_weights))
	print("Weights: " + str(starting_weights))

for i in range(num_trials):
	print("trial: " + str(i + starting_weights))
	cur_state = env.reset()
	action = env.action_space.sample()
	reward_sum = 0.
	last_action = env.state[5]
	for j in range(trial_len):
		#env.render()
		cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
		action = actor_critic.act(cur_state) + actor_noise()
		action = np.where(np.greater(np.abs(action), np.pi/2), (np.sign(action))*(np.abs(action)-np.pi), action)
		action = action.reshape((1, env.action_space.shape[0]))

		for k in range(5):
			env.state[5] = last_action
			new_state, reward, done, _ = env.step(action[0][0])

		last_action = action[0][0]
		reward_sum += reward
		if j == (trial_len - 1):
			print("reward sum: " + str(reward_sum))

		if done == True:
			print("reward sum: " + str(reward_sum))
			break

		actor_critic.train()
		actor_critic.update_target()
		
		new_state = new_state.reshape((1, env.observation_space.shape[0]))

		actor_critic.remember(cur_state, action, reward, new_state, done)
		cur_state = new_state

	if (i % 5 == 0):
		print("Render")
		cur_state = env.reset()
		reward_sum = 0.
		last_action = env.state[5]
		env.render()
		for j in range(600):
			cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
			action = actor_critic.act(cur_state)
			action = action.reshape((1, env.action_space.shape[0]))

			for k in range(5):
				env.state[5] = last_action
				new_state, reward, done, _ = env.step(action[0][0])
				env.render()

			last_action = action[0][0]
			reward_sum += reward
			if j == (600 - 1):
				print("reward: " + str(reward))
				print("reward sum: " + str(reward_sum))
				print("e_u: " + str(env.state[0] - env.target[2]))
				print("y_e: " + str(env.state[3]))
			
			if done == True:
				print("reward: " + str(reward))
				print("reward sum: " + str(reward_sum))
				print("e_u: " + str(env.state[0] - env.target[2]))
				print("y_e: " + str(env.state[3]))
				break

			new_state = new_state.reshape((1, env.observation_space.shape[0]))

			actor_critic.remember(cur_state, action, reward, new_state, done)
			cur_state = new_state

	if (i % 100 == 0 and i > 1):
		actor_critic.actor_model.save_weights("./ddpg_models/iteration" + str(i + starting_weights))
		actor_critic.critic_model.save_weights("./ddpg_models/critic/critic" + str(i + starting_weights))
		actor_critic.target_actor_model.save_weights("./ddpg_models/target_actor/target_actor" + str(i + starting_weights))
		actor_critic.target_critic_model.save_weights("./ddpg_models/target_critic/target_critic" + str(i + starting_weights))