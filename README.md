# CS4453B-GroupProject

A Comparative Study on Pong



For this version of pong the game ends when one player reaches 21 points. If the model is around -21 reward it means it doesn't know how to score yet. ie worst case scenario.



All code was done in Python 3.11.

The Command to install all packages is:

`py -m pip install "gymnasium[atari,accept-rom-license]==0.29.1" stable-baselines3 shimmy opencv-python tensorboard`



Tensorboard is used to visualise the model while it is training. To use it begin training with `tensorboard_log="./../pong_tensorboard/Scratch"` set in the model parameters. Then in a CMD window from the `CS4453B-GroupProject` folder run the command `tensorboard --logdir ./pong_tensorboard/Scratch`. Finally go to `http://localhost:6006/` and you can see live graphs of the models rewards and losses. When making the final versions of the models remove the `Scratch` from the above commands.