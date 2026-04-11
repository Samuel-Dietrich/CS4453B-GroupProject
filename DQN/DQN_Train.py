from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class DQNAgent(object):
    def __init__(self, log_dir):
        print("Setup Environment...")

        # Create the environment
        env_id = "PongNoFrameskip"
        env = make_atari_env(env_id, n_envs=10)

        # Stack 4 frames so the model can perceive velocity
        env = VecFrameStack(env, n_stack=4)

        print("Initialize DQN...")

        # Initialize the DQN model
        # 'CnnPolicy' is used because the input is images (pixels)
        self.model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=100_000,  # Experience Replay size (RAM intensive)
            learning_starts=10_000,  # Don't train until buffer has some data
            batch_size=32,  # Classic Atari DQN uses 32
            tau=1.0,  # Hard update for target network
            target_update_interval=1000,  # How often to update the target network
            train_freq=4,  # Update the model every 4 steps
            gradient_steps=1,  # How many gradient steps to do per update
            exploration_fraction=0.1,  # Fraction of total steps for epsilon decay
            exploration_final_eps=0.01,  # Final "greedy" exploration rate
            verbose=1,
            tensorboard_log="./../pong_tensorboard/Scratch"
        )

    def trainAgent(self, total_timesteps):
        # Train the agent
        print(f"Training started TimeSteps {total_timesteps}...")
        self.model.learn(total_timesteps=total_timesteps, tb_log_name=f"DQN_{total_timesteps / 1_000_000}_Step")

        self.model.save(f"dqn_pong_model_{total_timesteps}")
        print("Model saved!")
