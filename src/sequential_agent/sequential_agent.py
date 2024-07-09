import sys
import numpy as np

# https://flatland.aicrowd.com/challenges/flatland3/flatland-3-migration-guide.html

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
# from flatland.envs.schedule_generators import complex_schedule_generator
# schedule_generators are now renamed to line_generators
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from ordered_policy import OrderedPolicy

"""
This file shows how to move agents in a sequential way: it moves the trains one by one, following a shortest path strategy.
This is obviously very slow, but it's a good way to get familiar with the different Flatland components: RailEnv, TreeObsForRailEnv, etc...

multi_agent_training.py is a better starting point to train your own solution!
"""

np.random.seed(2)

n_agents = 1
x_dim = 35
y_dim = 35
n_cities = 2
max_rails_between_cities = 2
max_rail_pairs_in_city = 2
seed = 42

# Observation parameters
observation_tree_depth = 2
observation_radius = 10

# Exploration parameters
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.997  # for 2500ts

# x_dim = np.random.randint(8, 20)
# y_dim = np.random.randint(8, 20)
# n_agents = np.random.randint(3, 8)
# n_goals = n_agents + np.random.randint(0, 3)
# min_dist = int(0.75 * min(x_dim, y_dim))

env = RailEnv(
    width=x_dim,
    height=y_dim,
    rail_generator=sparse_rail_generator(
        max_num_cities=n_cities,
        grid_mode=False,
        max_rails_between_cities=max_rails_between_cities,
        max_rail_pairs_in_city=max_rail_pairs_in_city
    ),
    line_generator=sparse_line_generator(),
    obs_builder_object=TreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv()),
    number_of_agents=n_agents)
env.reset(True, True)

tree_depth = 1
observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PGL", )
handle = env.get_agent_handles()
n_episodes = 10
max_steps = 100 * (env.height + env.width)
record_images = False
policy = OrderedPolicy()
action_dict = dict()

for trials in range(1, n_episodes + 1):
    # Reset environment
    obs, info = env.reset(True, True)
    done = env.dones
    env_renderer.reset()
    frame_step = 0

    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

        if record_images:
            env_renderer.gl.save_image("./Images/flatland_frame_{:04d}.bmp".format(frame_step))
            frame_step += 1

        # Action
        acting_agent = 0
        for a in range(env.get_num_agents()):
            if done[a]:
                acting_agent += 1
            if a == acting_agent:
                action = policy.act(obs[a]).item() # changed to int
            else:
                action = 4
            action_dict.update({a: action})

        # Environment step
        obs, all_rewards, done, _ = env.step(action_dict)

        if done['__all__']:
            break