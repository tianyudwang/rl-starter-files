import argparse
import logging
import numpy as np

import utils



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

    # Create a window to view the environment
    env.render('human')
    
    for episode in range(args.episodes):
        obs = env.reset()
        
        # Reset expert
        grid = env.grid.encode()[:,:,0]
        agent.reset(grid, env.agent_pos)
        # Compute shortest path from all states to goal
        agent.shortest_path()

        while True:
            env.render('human')
    
            # Apply the first control
            control = agent.get_expert_control()
            import pdb; pdb.set_trace()
            # Cheat the action by turning the agent to the direction
            # of control and then step forward
            env.agent_dir = control
            forward = 2
            obs, reward, done, _ = env.step(forward)
            # Update agent position 
            agent.update_agent_pos(env.agent_pos)
    
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
    args = parser.parse_args()
    main()
