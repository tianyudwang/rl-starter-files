import sys, os
import argparse
import logging
import numpy as np

import utils

def collect_expert_demo(filename, obs, state, control, goal):
    """
    Collects expert demonstration
    observation, state, control, goal
    """
    
    lidar_points = obs['lidar_points']
    semantic_labels = obs['semantic_labels']

    np.savez_compressed(
        filename,
        state=np.array(state),
        control=np.array(control),
        lidar_points=np.array(lidar_points),
        semantic_labels=np.array(semantic_labels),
        goal=np.array(goal)
    )

def main():

    logging.basicConfig(level=logging.INFO)

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Load environment
    env = utils.make_env(args.env, args.seed)
    for _ in range(args.shift):
        env.reset()
    logging.info("Environment loaded")

    # Load agent
    agent = utils.ExpertAgent()
    logging.info("Agent loaded")

    # Create a window to view the environment
    env.render('human')
    
    for episode in range(args.episodes):
        # Set save directory
        traj_dir = os.path.join(args.save_dir, 'traj_{0:03d}'.format(episode))
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir, exist_ok=False)
        logging.info("Beginning expert episode {}".format(episode))

        # Reset environment and expert
        obs = env.reset() 
        grid = env.grid.encode()[:,:,0]
        agent.reset(grid, env.agent_pos)

        # Compute shortest path from all states to goal
        agent.shortest_path()
        logging.info("Episode {} shortest path exists".format(episode))
        
        # skip episode if shortest path is not feasible
        if not agent.get_expert_controls():
            logging.warning("Episode {} does not have infeasible path".format(episode))
            continue
        
        t = 0
        while True:
            env.render('human')
    
            # Apply the first control
            control = agent.get_expert_control()

            # Collect and save expert demonstration
            collect_expert_demo(
                os.path.join(traj_dir, 'step_{0:03d}'.format(episode, t)),
                env.gen_obs(), agent.agent_pos, 
                control, agent.goal_pos
            )

            # Cheat the action by turning the agent 
            # to the direction of control and then step forward
            env.agent_dir = control
            forward = 2
            obs, reward, done, _ = env.step(forward)
            # Update agent position 
            agent.update_agent_pos(env.agent_pos)
    
            t += 1

            if done or env.window.closed:
                break
    
        if env.window.closed:
            break
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, default='MiniGrid-RandomLavaS9-v0',
        help='name of the environment to be run')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed (default: 0)')
    parser.add_argument(
        '--shift', type=int, default=0,
        help='number of times the environment is reset at the beginning (default: 0)')
    parser.add_argument(
        '--episodes', type=int, default=100,
        help='number of episodes to collect expert data')
    parser.add_argument(
        '--save_dir', type=str, 
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/',
        help='directory to save the expert demonstrations')
    args = parser.parse_args()
    main()
