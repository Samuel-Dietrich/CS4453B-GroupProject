from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

print("Setup Environment...")

# Create the environment
env_id = "PongNoFrameskip"
env = make_atari_env(env_id, n_envs=10, seed=0)

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

print("Initialize A2C...")

# Initialize the A2C model
model = A2C(
    policy="CnnPolicy",
    env=env,
    learning_rate=7e-4,
    n_steps=128,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./../pong_tensorboard/Scratch"
)

# 4. Train the agent
print("Training started...")
TOTALSTEPS = 500_000
model.learn(total_timesteps=TOTALSTEPS, tb_log_name=f"A2C_{TOTALSTEPS/1_000_000}_Step")

# 5. Save the model
model.save(f"a2c_pong_model_{TOTALSTEPS}")
print("Model saved!")