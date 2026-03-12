import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Load the trained model
MODELTIMESTEPS = 1_000_000
model = PPO.load(f"ppo_pong_model_{MODELTIMESTEPS}")
env_id = "PongNoFrameskip"

# Create a human-viewable environment
# Use env_kwargs to pass render_mode to the underlying Gymnasium env
env = make_atari_env(
    env_id,
    n_envs=1,
    seed=0,
    env_kwargs={"render_mode": "human"}
)
env = VecFrameStack(env, n_stack=4)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.01) # Slow it down so you can watch
    if dones:
        obs = env.reset()