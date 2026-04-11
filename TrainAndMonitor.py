from DQN.DQN_Train import DQNAgent
from PPO.PPO_Train import PPOAgent
from A2C.A2C_Train import A2CAgent
import psutil
import threading
import time
from torch.utils.tensorboard import SummaryWriter

TOTAL_TIMESTEPS = 100_000
# make sure the 'tensorboard_log' path in the agent is the same here
LOG_DIR = f"./pong_tensorboard/Scratch"

"""
TO CHANGE MODEL CHANGE THE BELOW 2 LINES TO ONE OF THE FOLLOWING
MODEL_NAME: Change first 3 letters in the string
            - Options are DQN, A2C and PPO
            - Results in DQN_{TOTAL_TIMESTEPS / 1_000_000}_Step_1, A2C_{TOTAL_TIMESTEPS / 1_000_000}_Step_1 or PPO_{TOTAL_TIMESTEPS / 1_000_000}_Step_1
Agent: Change first 3 letters in the class name
            -  Options are DQN, A2C and PPO.
            -  Results in DQNAgent, A2CAgent or PPOAgent

"""
MODEL_NAME = f"DQN_{TOTAL_TIMESTEPS / 1_000_000}_Step_3" # dont forget the _1 at the end which does not show in the agent
Agent = DQNAgent("./." + LOG_DIR)



thread1 = threading.Thread(target=Agent.trainAgent, args=(TOTAL_TIMESTEPS,))
writer = SummaryWriter(log_dir=LOG_DIR + "/" + MODEL_NAME)

thread1.start()

step = 0
try:
    while thread1.is_alive():
        # Get metrics
        memory_percent = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)

        # Write to TensorBoard
        writer.add_scalar("System/Memory_Usage", memory_percent, step)
        writer.add_scalar("System/CPU_Usage", cpu_usage, step)

        print(f"Logged - CPU: {cpu_usage}%, MEM: {memory_percent}%")

        step += 1
        time.sleep(10)

finally:
    writer.close()
    thread1.join()