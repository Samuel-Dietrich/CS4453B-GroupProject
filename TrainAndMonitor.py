from DQN.DQN_Train import DQNAgent
from PPO.PPO_Train import PPOAgent
from A2C.A2C_Train import A2CAgent
import psutil
from stable_baselines3.common.callbacks import BaseCallback

TOTAL_TIMESTEPS = 10_000_000
# make sure the 'tensorboard_log' path in the agent is the same here
LOG_DIR = f"./../pong_tensorboard"


class SystemMetricsCallback(BaseCallback):
    last_step = -1001
    def __init__(self, verbose=0):
        super(SystemMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls >= self.last_step + 1000:
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent

            self.logger.record("system/cpu_usage", cpu_usage)
            self.logger.record("system/memory_usage", mem_usage)
            self.last_step = self.n_calls

            print(f"Logged - CPU: {cpu_usage}%, MEM: {mem_usage}%")

        return True

"""
TO CHANGE MODEL CHANGE THE BELOW 2 LINES TO ONE OF THE FOLLOWING
MODEL_NAME: Change first 3 letters in the string
            - Options are DQN, A2C and PPO
            - Results in DQN_{TOTAL_TIMESTEPS / 1_000_000}_Step, A2C_{TOTAL_TIMESTEPS / 1_000_000}_Step or PPO_{TOTAL_TIMESTEPS / 1_000_000}_Step
Agent: Change first 3 letters in the class name
            -  Options are DQN, A2C and PPO.
            -  Results in DQNAgent, A2CAgent or PPOAgent

"""
MODEL_NAME = f"PPO_{TOTAL_TIMESTEPS / 1_000_000}_Step"
Agent = PPOAgent(LOG_DIR)

sys_callback = SystemMetricsCallback()
Agent.trainAgent(TOTAL_TIMESTEPS, MODEL_NAME, sys_callback)
