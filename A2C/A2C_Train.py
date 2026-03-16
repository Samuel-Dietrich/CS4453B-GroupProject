import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

print("Setup Environment...")

# 1. Create the environment
# NOTE: Standard DQN usually trains on a single environment instance
# because it is off-policy, though SB3 supports vectorized DQN.
env_id = "PongNoFrameskip"
env = make_atari_env(env_id, n_envs=10, seed=0)

# 2. Stack 4 frames
# This is crucial for DQN so it can see the direction/speed of the ball
env = VecFrameStack(env, n_stack=4)

print("Initialize A2C...")

# 3. Initialize the DQN model
# We use 'CnnPolicy' for pixel input.
model = A2C(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,         # Usually lower than PPO for stability
    verbose=1,
    tensorboard_log="./a2c_pong_tensorboard/"
)

# 4. Train the agent
print("Training started...")
TOTALSTEPS = 500_000
model.learn(total_timesteps=TOTALSTEPS, log_interval=10)

# 5. Save the model
model.save(f"a2c_pong_model_{TOTALSTEPS}")
print("Model saved!")