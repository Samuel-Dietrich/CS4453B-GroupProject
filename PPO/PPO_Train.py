from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

class PPOAgent(object):
    def __init__(self, log_dir):
        print("Setup Environment...")

        # Create the environment
        # Change n_envs to the number of cores your processor has
        env_id = "PongNoFrameskip"
        env = make_atari_env(env_id, n_envs=10)

        # Stack 4 frames so the model can perceive velocity
        env = VecFrameStack(env, n_stack=4)

        print("Initialize PPO...")

        # Initialize the PPO model
        # 'CnnPolicy' is used because the input is images (pixels)
        self.model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=2.5e-4,  # Standard learning rate
            n_steps=128,  # Steps to run per update per env
            batch_size=256,  # Minibatch size for gradient updates
            n_epochs=4,  # Number of times to reuse collected data
            clip_range=0.1,  # Tightens the PPO update for stability
            ent_coef=0.01,  # Keeps paddle moving/exploring
            vf_coef=0.5,  # How much the "Critic" matters
            verbose=1,
            tensorboard_log=log_dir # Log location for comparison
        )
    def trainAgent(self, total_timesteps, model_name, callback):

        # Train the agent
        print("Training started...")
        self.model.learn(total_timesteps=total_timesteps, tb_log_name=model_name, callback=callback)

        self.model.save(f"ppo_pong_model_{total_timesteps}")
        print("Model saved!")