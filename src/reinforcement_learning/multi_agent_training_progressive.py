from utils.timer import Timer
from utils.observation_utils import normalize_observation
from reinforcement_learning.deep_policy import (
    DQN,
    DoubleDQN,
    DuelingDQN,
    DoubleDuelingDQN,
    SARSA,
    ExpectedSARSA,
)
from envs import ENV_PARAMS
from datetime import datetime
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import psutil
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import (
    ParamMalfunctionGen,
    MalfunctionParameters,
)
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


"""
This file shows how to train multiple agents using
a reinforcement learning approach.
After training an agent, you can submit it straight
away to the Flatland 3 challenge!

Agent documentation:
https://flatland.aicrowd.com/tutorials/rl/multi-agent.html
Submission documentation:
https://flatland.aicrowd.com/challenges/flatland3/first-submission.html
"""


def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rail_pairs_in_city = env_params.max_rail_pairs_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50,
    )

    return RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed,
    )


def train_agent(
    train_params, train_env_params_list, eval_env_params, obs_params
):
    try:
        import wandb

        # runname = 'flatland-rl_run123' # specify your run name
        # if not runname or runname == 'flatland-rl_run123':
        #     runname = 'flatland-rl_run_' +
        # datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{train_params.policy}_highest_env_"\
            f"{train_params.highest_level}_obstreedepth"\
            f"_{train_params.obstreedepth}_hs_{train_params.hidden_size}"
        if len(train_env_params_list) == 1:
            name = f"{train_params.policy}_env_"\
                f"{train_params.training_env_config}_obstreedepth"\
                f"_{train_params.obstreedepth}_hs_{train_params.hidden_size}"

        wandb.init(
            mode="offline",  # specify if you want to log to W&B 'disabled',
            # 'online' or 'offline' (offline logs to local file)
            sync_tensorboard=True,
            name=name,
            project="Reinforce_TrainReScheduling-reinforcement_learning",
        )

    except ImportError:
        print("Install wandb to log to Weights & Biases")

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime("%y%m%d%H%M%S")

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(
        obs_params.observation_max_path_depth
    )
    tree_observation = TreeObsForRailEnv(
        max_depth=obs_params.observation_tree_depth, predictor=predictor
    )

    # Calculate the state size given the depth of the
    # tree observation and the number of features
    train_env = create_rail_env(train_env_params_list[0], tree_observation)
    eval_env = create_rail_env(eval_env_params, tree_observation)

    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum(
        [np.power(4, i) for i in range(obs_params.observation_tree_depth + 1)]
    )  # level 0 = 4**0, level 1
    state_size = n_features_per_node * n_nodes
    action_size = 5  # The action space of flatland is 5 discrete actions

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    if train_params.policy == "dqn":
        policy = DQN(state_size, action_size, train_params)
    elif train_params.policy == "double_dqn":
        policy = DoubleDQN(state_size, action_size, train_params)
    elif train_params.policy == "dueling_dqn":
        policy = DuelingDQN(state_size, action_size, train_params)
    elif train_params.policy == "double_dueling_dqn":
        policy = DoubleDuelingDQN(state_size, action_size, train_params)
    elif train_params.policy == "sarsa":
        policy = SARSA(state_size, action_size, train_params)
    elif train_params.policy == "expected_sarsa":
        policy = ExpectedSARSA(
            state_size,
            action_size,
            train_params,
            evaluation_mode=False,
            expected_sarsa_temperature=train_params.expected_sarsa_temperature,
        )

    # Load existing replay buffer
    if train_params.restore_replay_buffer:
        try:
            policy.load_replay_buffer(train_params.restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print(
                "\n🛑 Could't load replay buffer, " //
                "were the experiences generated using the same tree depth?"
            )
            print(e)
            exit(1)

    print(
        "\n💾 Replay buffer status: {}/{} experiences".format(
            len(policy.memory.memory), train_params.buffer_size
        )
    )
    hdd = psutil.disk_usage("/")
    if train_params.save_replay_buffer and (hdd.free / (2**30)) < 500.0:
        print(
            "⚠️  Careful! Saving replay buffers will quickly" //
            " consume a lot of disk space. You have {:.2f}gb left.".format(
                hdd.free / (2**30)
            )
        )

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(eval_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()

    for env_idx, train_env_params in enumerate(train_env_params_list):

        # Set the seeds
        random.seed(train_env_params.seed)
        train_env = create_rail_env(train_env_params, tree_observation)
        if train_params.render:
            env_renderer = RenderTool(train_env, gl="PGL")

        print(
            "\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}' Env {}/{}.\n".format(  # noqa: E501
                train_env_params.n_agents,
                train_env_params.x_dim,
                train_env_params.y_dim,
                train_params.n_episodes,
                train_params.n_evaluation_episodes,
                train_params.checkpoint_interval,
                training_id,
                env_idx + 1,
                (
                    train_params.highest_level + 1
                    if train_params.highest_level is not None
                    else 1
                ),
            )
        )
        episode_idx = 0
        for local_episode_idx in range(train_params.n_episodes):
            episode_idx += 1

            step_timer = Timer()
            reset_timer = Timer()
            learn_timer = Timer()
            preproc_timer = Timer()
            inference_timer = Timer()

            # Reset environment
            reset_timer.start()
            obs, info = train_env.reset(
                regenerate_rail=True, regenerate_schedule=True
            )
            reset_timer.end()

            # Init these values after reset()
            max_steps = train_env._max_episode_steps
            action_count = [0] * action_size
            action_dict = dict()
            agent_obs = [None] * train_env_params.n_agents
            agent_prev_obs = [None] * train_env_params.n_agents
            agent_prev_action = [2] * train_env_params.n_agents
            update_values = [False] * train_env_params.n_agents

            if train_params.render:
                env_renderer.set_new_rail()

            score = 0
            nb_steps = 0
            actions_taken = []

            # Build initial agent-specific observations
            for agent in train_env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(
                        obs[agent],
                        obs_params.observation_tree_depth,
                        observation_radius=obs_params.observation_radius,
                    )
                    agent_prev_obs[agent] = agent_obs[agent].copy()

            # Run episode
            for step in range(max_steps):
                inference_timer.start()
                for agent in train_env.get_agent_handles():
                    if info["action_required"][agent]:
                        update_values[agent] = (
                            True  # only learn from timesteps
                            # where somethings happened
                        )
                        action = policy.act(
                            agent_obs[agent], eps=train_params.eps_start
                        )

                        action_count[action] += 1
                        actions_taken.append(action)
                    else:
                        # An action is not required if the train hasn't
                        # joined the railway network,
                        # if it already reached its target, or if is
                        # currently malfunctioning.
                        update_values[agent] = (
                            False  # only learn from timesteps where
                            # somethings happened
                        )
                        action = 0

                    action_dict.update({agent: action})
                inference_timer.end()

                # Environment step
                step_timer.start()
                next_obs, all_rewards, done, info = train_env.step(action_dict)
                step_timer.end()

                # Render an episode at some interval
                if (
                    train_params.render
                    and episode_idx % train_params.checkpoint_interval == 0
                ):
                    env_renderer.render_env(
                        show=True,
                        frames=False,
                        show_observations=False,
                        show_predictions=False,
                    )

                # Update replay buffer and train agent
                for agent in train_env.get_agent_handles():
                    if update_values[agent] or done["__all__"]:
                        # Only learn from timesteps where somethings happened
                        learn_timer.start()
                        policy.step(
                            agent_prev_obs[agent],
                            agent_prev_action[agent],
                            all_rewards[agent],
                            agent_obs[agent],
                            action_dict[agent],
                            done[agent],
                        )
                        learn_timer.end()

                        agent_prev_obs[agent] = agent_obs[agent].copy()
                        agent_prev_action[agent] = action_dict[agent]

                    # Preprocess the new observations
                    if next_obs[agent]:
                        preproc_timer.start()
                        agent_obs[agent] = normalize_observation(
                            next_obs[agent],
                            obs_params.observation_tree_depth,
                            observation_radius=obs_params.observation_radius,
                        )
                        preproc_timer.end()

                    score += all_rewards[agent]

                nb_steps = step

                if done["__all__"]:
                    break

            # Epsilon decay
            train_params.eps_start = max(
                train_params.eps_end,
                train_params.eps_decay * train_params.eps_start,
            )

            # Collect information about training
            tasks_finished = sum(
                [agent.state == TrainState.DONE for agent in train_env.agents]
            )
            completion = tasks_finished / max(1, train_env.get_num_agents())
            normalized_score = score / (
                max_steps * train_env.get_num_agents()
            )  #

            if np.sum(action_count) > 0:
                action_probs = action_count / np.sum(action_count)
            else:
                action_probs = action_count
            action_count = [1] * action_size

            if not actions_taken:
                actions_taken = [0]

            smoothing = 0.99
            smoothed_normalized_score = (
                smoothed_normalized_score * smoothing
                + normalized_score * (1.0 - smoothing)
            )
            smoothed_completion = (
                smoothed_completion * smoothing
                + completion * (1.0 - smoothing)
            )

            if episode_idx % train_params.checkpoint_interval == 0:
                torch.save(
                    policy.qnetwork_local,
                    f"./checkpoints/{train_params.policy}"
                    + training_id
                    + "-"
                    + str(episode_idx)
                    + ".pth",
                )

                if train_params.save_replay_buffer:
                    policy.save_replay_buffer(
                        f"./replay_buffers/{train_params.policy}"
                        + training_id
                        + "-"
                        + str(episode_idx)
                        + ".pkl"
                    )

                if train_params.render:
                    env_renderer.close_window()

            print(
                "\r🚂 Ep {}"
                "\t 🏆 Score: {:.3f}"
                " Avg: {:.3f}"
                "\t Done: {:.2f}%"
                " Avg: {:.2f}%"
                "\t 🎲 ϵ: {:.3f} "
                "\t 🔀 Action Prob.: {}"
                "\t ⏱️ Rst {:.3f}s"
                "\t Step {:.3f}s"
                "\t Lrn {:.3f}s"
                "\t Preproc {:.3f}s"
                "\t Tot {:.3f}s".format(
                    episode_idx,
                    normalized_score,
                    smoothed_normalized_score,
                    100 * completion,
                    100 * smoothed_completion,
                    train_params.eps_start,
                    format_action_prob(action_probs),
                    reset_timer.get(),
                    step_timer.get(),
                    learn_timer.get(),
                    preproc_timer.get(),
                    training_timer.get_current(),
                ),
                end=" ",
            )

            if (
                episode_idx % train_params.checkpoint_interval == 0
                and train_params.n_evaluation_episodes > 0
            ):
                scores, completions, nb_steps_eval = eval_policy(
                    eval_env, policy, train_params, obs_params
                )

                wandb.log(
                    {
                        "evaluation/scores_min": np.min(scores),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/scores_max": np.max(scores),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/scores_mean": np.mean(scores),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/scores_std": np.std(scores),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/scores": np.array(scores),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/completions_min": np.min(completions),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/completions_max": np.max(completions),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/completions_mean": np.mean(completions),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/completions_std": np.std(completions),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/completions": np.array(completions),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/nb_steps_min": np.min(nb_steps_eval),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/nb_steps_max": np.max(nb_steps_eval),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/nb_steps_mean": np.mean(nb_steps_eval),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/nb_steps_std": np.std(nb_steps_eval),
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/nb_steps": wandb.Histogram(
                            np_histogram=np.histogram(np.array(nb_steps_eval))
                        ),
                        "episode": episode_idx,
                    }
                )

                smoothing = 0.9
                smoothed_eval_normalized_score = (
                    smoothed_eval_normalized_score * smoothing
                    + np.mean(scores) * (1.0 - smoothing)
                )
                smoothed_eval_completion = (
                    smoothed_eval_completion * smoothing
                    + np.mean(completions) * (1.0 - smoothing)
                )
                wandb.log(
                    {
                        "evaluation/smoothed_score":
                        smoothed_eval_normalized_score,
                        "step": episode_idx,
                    }
                )
                wandb.log(
                    {
                        "evaluation/smoothed_completion":
                        smoothed_eval_completion,
                        "step": episode_idx,
                    }
                )

            wandb.log(
                {"training/score": normalized_score, "step": episode_idx}
            )
            wandb.log(
                {
                    "training/smoothed_score": smoothed_normalized_score,
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "training/completion": np.mean(completion),
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "training/smoothed_completion": np.mean(
                        smoothed_completion
                    ),
                    "step": episode_idx,
                }
            )
            wandb.log({"training/nb_steps": nb_steps, "step": episode_idx})
            wandb.log(
                {
                    "actions/distribution": np.array(actions_taken),
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "actions/nothing": action_probs[RailEnvActions.DO_NOTHING],
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "actions/left": action_probs[RailEnvActions.MOVE_LEFT],
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "actions/forward": action_probs[
                        RailEnvActions.MOVE_FORWARD
                    ],
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "actions/right": action_probs[RailEnvActions.MOVE_RIGHT],
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "actions/stop": action_probs[RailEnvActions.STOP_MOVING],
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "training/epsilon": train_params.eps_start,
                    "step": episode_idx,
                }
            )
            wandb.log(
                {
                    "training/buffer_size": len(policy.memory),
                    "step": episode_idx,
                }
            )
            wandb.log({"training/loss": policy.loss, "step": episode_idx})
            wandb.log({"timer/reset": reset_timer.get(), "step": episode_idx})
            wandb.log({"timer/step": step_timer.get(), "step": episode_idx})
            wandb.log({"timer/learn": learn_timer.get(), "step": episode_idx})
            wandb.log(
                {"timer/preproc": preproc_timer.get(), "step": episode_idx}
            )
            wandb.log(
                {
                    "timer/total": training_timer.get_current(),
                    "step": episode_idx,
                }
            )


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        max_steps = env._max_episode_steps
        action_dict = dict()
        agent_obs = [None] * env.get_num_agents()

        score = 0.0

        final_step = 0

        for step in range(max_steps):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(
                        obs[agent],
                        tree_depth=tree_depth,
                        observation_radius=observation_radius,
                    )

                action = 0
                if info["action_required"][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done["__all__"]:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(
            [agent.state == TrainState.DONE for agent in env.agents]
        )
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print(
        "\t✅ Eval: score {:.3f} done {:.1f}%".format(
            np.mean(scores), np.mean(completions) * 100.0
        )
    )

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_episodes",
        help="number of episodes to run",
        default=2500,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--training_env_config",
        help="training config id (eg 0 for Test_0)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--evaluation_env_config",
        help="evaluation config id (eg 0 for Test_0)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--n_evaluation_episodes",
        help="number of evaluation episodes",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--checkpoint_interval",
        help="checkpoint interval",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--eps_start", help="max exploration", default=1.0, type=float
    )
    parser.add_argument(
        "--eps_end", help="min exploration", default=0.01, type=float
    )
    parser.add_argument(
        "--eps_decay", help="exploration decay", default=0.99, type=float
    )
    parser.add_argument(
        "--buffer_size", help="replay buffer size", default=int(1e5), type=int
    )
    parser.add_argument(
        "--buffer_min_size",
        help="min buffer size to start training",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--restore_replay_buffer",
        help="replay buffer to restore",
        default="",
        type=str,
    )
    parser.add_argument(
        "--save_replay_buffer",
        help="save replay buffer at each evaluation interval",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--batch_size", help="minibatch size", default=128, type=int
    )
    parser.add_argument(
        "--gamma", help="discount factor", default=0.99, type=float
    )
    parser.add_argument(
        "--tau",
        help="soft update of target parameters",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--learning_rate", help="learning rate", default=0.5e-4, type=float
    )
    parser.add_argument(
        "--hidden_size",
        help="hidden size (2 fc layers)",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--update_every",
        help="how often to update the network",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--use_gpu", help="use GPU if available", default=False, type=bool
    )
    parser.add_argument(
        "--num_threads",
        help="number of threads PyTorch can use",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--render", help="render 1 episode in 100", default=False, type=bool
    )
    parser.add_argument(
        "--policy",
        help="Policy to use: options: dqn, double_dqn, dueling_dqn," //
        " double_dueling_dqn, sarsa, expected_sarsa",
        type=str,
    )
    parser.add_argument(
        "--obstreedepth", help="depth of obs tree", default=2, type=int
    )
    parser.add_argument(
        "--expected_sarsa_temperature",
        help="temperature for learning Q value",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--highest_level", help="highest training environment level", type=int
    )  # New optional parameter
    parser.add_argument(
        "--n_step",
        help="Number of reward steps to accumulate (e.g for n-step sarsa)",
        default=1,
        type=int,
    )
    training_params = parser.parse_args()

    env_params = ENV_PARAMS

    obs_params = {
        "observation_tree_depth": training_params.obstreedepth,  # default is 2
        "observation_radius": 10,  # normalization constant for normalizing
        # observations to make rf more stable across runs
        # (not actually a radius)
        "observation_max_path_depth": 30,  # is used in the path predictor and
        # ultimately to add info for agent a where b and c... will probably be,
        # the predictor here only returns n locations of the
        # predicted agent's b and c... position
    }

    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print(
                "\n🛑 Invalid environment configuration, only Test_0 to Test_"
                + "{} are supported.".format(
                    len(env_params) - 1
                )
            )
            exit(1)

    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    if training_params.highest_level is not None:
        check_env_config(
            training_params.highest_level
        )  # Check highest level validity

        # Get list of training environments for progressive training
        training_env_params_list = [
            Namespace(**env_params[i])
            for i in range(training_params.highest_level + 1)
        ]

        print(
            "\nTraining environment parameters up to Test_{}:".format(
                training_params.highest_level
            )
        )
        for i, params in enumerate(training_env_params_list):
            print(f"Test_{i}:")
            pprint(vars(params))
    else:
        # Default to single environment training
        training_env_params_list = [
            Namespace(**env_params[training_params.training_env_config])
        ]

        print(
            "\nTraining environment parameters (Test_{}):".format(
                training_params.training_env_config
            )
        )
        pprint(vars(training_env_params_list[0]))

    evaluation_env_params = Namespace(
        **env_params[training_params.evaluation_env_config]
    )

    print(
        "\nEvaluation environment parameters (Test_{}):".format(
            training_params.evaluation_env_config
        )
    )
    pprint(vars(evaluation_env_params))
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)

    print("\n🚂 Training policy: {}".format(training_params.policy))
    train_agent(
        training_params,
        training_env_params_list,
        evaluation_env_params,
        Namespace(**obs_params),
    )
