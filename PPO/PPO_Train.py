import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

print("Setup Environment...")

# 1. Create the environment
# change n_envs to the number of cores ur computer has (Sams is 6 preformance so thats y it is 6)
env_id = "PongNoFrameskip"
env = make_atari_env(env_id, n_envs=6, seed=0)

# 2. Stack 4 frames so the model can perceive velocity
env = VecFrameStack(env, n_stack=4)

print("Initialize PPO...")

# 3. Initialize the PPO model
# 'CnnPolicy' is used because the input is images (pixels)
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=2.5e-4,  # Standard "Atari" learning rate
    n_steps=128,  # Steps to run per update per env
    batch_size=256,  # Minibatch size for gradient updates
    n_epochs=4,  # Number of times to reuse collected data
    clip_range=0.1,  # Tightens the PPO update for stability
    ent_coef=0.01,  # Keeps paddle moving/exploring
    vf_coef=0.5,  # How much the "Critic" matters
    verbose=1,
    tensorboard_log="./ppo_pong_tensorboard/"
)

# 4. Train the agent
print("Training started...")
TOTALSTEPS = 1_000_000
model.learn(total_timesteps=TOTALSTEPS)

# 5. Save the model
model.save(f"ppo_pong_model_{TOTALSTEPS}")
print("Model saved!")