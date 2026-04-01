import psutil
from DQN.DQN_Train import DQNAgent
from PPO.PPO_Train import PPOAgent
from A2C.A2C_Train import A2CAgent
import threading
import time

TOTAL_TIMESTEPS = 10_000


Agent = DQNAgent()



thread1 = threading.Thread(target=Agent.trainAgent, args=(TOTAL_TIMESTEPS,))
thread1.start()

# Main loop while thread is running
mem = []
cpu = []

while thread1.is_alive():
    memory_percent = psutil.virtual_memory().percent
    mem.append(memory_percent)
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu.append(cpu_usage)


    print(f"Percentage Used: {memory_percent}%")
    print(f"Current CPU Usage: {cpu_usage}%")
    time.sleep(60)
