import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

print("Setup Environment...")

# 1. Create the environment
# NOTE: Standard DQN usually trains on a single environment instance
# because it is off-policy, though SB3 supports vectorized DQN.
env_id = "PongNoFrameskip"
env = make_atari_env(env_id, n_envs=1, seed=0)

# 2. Stack 4 frames
# This is crucial for DQN so it can see the direction/speed of the ball
env = VecFrameStack(env, n_stack=4)

print("Initialize DQN...")

# 3. Initialize the DQN model
# We use 'CnnPolicy' for pixel input.
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,         # Usually lower than PPO for stability
    buffer_size=100_000,        # Experience Replay size (RAM intensive)
    learning_starts=10_000,     # Don't train until buffer has some data
    batch_size=32,              # Classic Atari DQN uses 32
    tau=1.0,                    # Hard update for target network
    target_update_interval=1000, # How often to update the target network
    train_freq=4,               # Update the model every 4 steps
    gradient_steps=1,           # How many gradient steps to do per update
    exploration_fraction=0.1,   # Fraction of total steps for epsilon decay
    exploration_final_eps=0.01, # Final "greedy" exploration rate
    verbose=1,
    tensorboard_log="./dqn_pong_tensorboard/"
)

# 4. Train the agent
print("Training started...")
TOTALSTEPS = 2_500_000
model.learn(total_timesteps=TOTALSTEPS, log_interval=10)

# 5. Save the model
model.save(f"dqn_pong_model_{TOTALSTEPS}")
print("Model saved!")