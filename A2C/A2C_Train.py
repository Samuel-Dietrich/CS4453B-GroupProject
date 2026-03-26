import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from typing import Callable

# Linear learning rate scheduler
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

print("Setup Environment...")

# Create the environment
env_id = "PongNoFrameskip-v4"
env = make_atari_env(env_id, n_envs=16, seed=0)

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

print("Initialize A2C...")

# Initialize the A2C model
model = A2C(
    policy="CnnPolicy",
    env=env,
    learning_rate=linear_schedule(7e-4),
    n_steps=5, 
    ent_coef=0.01,
    vf_coef=0.25,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./pong_tensorboard/Scratch",
    policy_kwargs=dict(
        optimizer_class=th.optim.RMSprop,
        optimizer_kwargs=dict(alpha=0.99, eps=1e-5, weight_decay=0)
    )
)

# Train the agent
print("Training started...")
TOTALSTEPS = 10_000_000
# log_interval indicates number of updates, now updates are much more frequent
model.learn(total_timesteps=TOTALSTEPS, tb_log_name=f"A2C_{TOTALSTEPS/1_000_000}_Step", log_interval=100)

# Save the model
model.save(f"./A2C/a2c_pong_model_{TOTALSTEPS}")
print("Model saved!")