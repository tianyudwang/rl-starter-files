import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import os
import utils

import numpy as np

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True, default='IRL',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.5,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

parser.add_argument("--save_dir", type=str, default='../grid_data',
                    help="Directory to save expert trajectories")


args = parser.parse_args()

def get_control(prev_state, curr_state):
    """
    retrieves the control from the previous state to the current state
    down = 0, right = 1, up = 2, left = 3 
    """
    assert(prev_state[0] != curr_state[0] or prev_state[1] != curr_state[1])

    du = curr_state[0] - prev_state[0]
    dv = curr_state[1] - prev_state[1]

    assert(np.abs(du) + np.abs(dv) <= 1)

    if du == 1 and dv == 0:
        control = 0
    if du == 0 and dv == 1:
        control = 1
    if du == 0 and dv == -1:
        control = 2
    if du == -1 and dv == 0:
        control = 3

    return control

def save_data(prev_state, curr_state, prev_obs, filename):
    """
    Save expert trajectory data in minigrid environment
    """

    control = get_control(prev_state, curr_state)
    goal = np.array([8, 8])

    np.savez_compressed(
        filename,
        state=np.array(prev_state),
        control=np.array(control),
        lidar_points=prev_obs['lidar_points'],
        semantic_labels=prev_obs['semantic_labels'],
        goal=np.array(goal)
        )


# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")


# Run the agent

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):

    save_folder = '/home/erl/rl-starter-files/data/traj_{:03d}'.format(episode)
    if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    obs = env.reset()

    curr_state = env.agent_pos

    frame = 0
    while True:
        env.render('human')

        prev_state = env.agent_pos
        prev_obs = obs
        action = agent.get_action(obs)

        # step the environment to the next state
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        curr_state = env.agent_pos

        # save data only when agent has moved
        if action == 2:
            filename = os.path.join(save_folder, 'frame_{:03d}'.format(frame))
            save_data(prev_state, curr_state, prev_obs, filename)
            frame += 1


        if done or env.window.closed:
            break

    if env.window.closed:
        break


