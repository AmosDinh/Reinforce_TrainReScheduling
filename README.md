# Reinforce_TrainReScheduling

If there are errors in the execution please use the commit 7654dff.

# Goal
This repository represents an attempt at tackeling the Vehicle Rescheduling Problem as modeled by the [flatland challenge](https://www.aicrowd.com/challenges/flatland) 
We seek to solve the partial problem of minimizing the time it takes to bring all the agents to their respective targets.


# Usage
    git clone https://github.com/AmosDinh/Reinforce_TrainReScheduling
    cd Reinfore_TrainReScheduling

    conda create -n rl python=3.8

    conda activate rl

    pip install -r requirements.txt

    cd src 

    python execute_local_sweep.py sweeps/sweep_policies.yaml

# Visualize a checkpoint
    python reinforcement_learning/multi_agent_training.py --n_episodes=10 --hidden_size=512 --buffer_size=128 --training_env_config=15 --policy="sarsa" --obstreedepth=2 --checkpoint="checkpoints/sweep_actual_final_sarsa_dqn_sarsa_env_0_obstreedepth_2_hs_512_nstep_1_gamma_0.99240710160404-14200.pth" --render=True --renderspeed=100

# Other sweeps
Sweeps (`src/sweeps/*`) contain hyperparamter grid searches. We use it with Weights & Biases
 (https://wandb.ai/) for online tracking but the sweeps can be run locally with `python src/execute_local_sweep.py sweeps/my_sweep_file.yaml`.
 ## Sweep files:
 - `sweep_baseline_shortest_path.yaml` contains the shortest path baseline
 - `sweep_expected_sarsa.yaml` contains the expected sarsa algorithm, --expected_sarsa_temperature can be varied
 - `sweep_n_episodes.yaml` checks influence of training length
 - `sweep_n_step_sarsa.yaml` checks n-step influence using SARSA
 - `sweep_policies.yaml` sweeps over all reinforcement learning methods
 - `sweep_progressive_incremental.yaml` trains with incrementally increasing number of agents
 - `sweep_progressive.yaml` trains successively on the 3 env configurations also used in evaluation
 - `sweep_sarsa_final.yaml` is the finally trained sarsa
 - `sweep_sarsa_gnn.yaml` is an experiment using a custom graph observation
 - `sweep_tree_depth.yaml` checks influence of observation depth on performance 

# (Not required) Weights and Biases usage:
Set wandb key on linux: WANDB_API_KEY=YOUR_API_KEY 
Our tracked metrics: [https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling](https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling)
<br>


# Files:
1. In file `reinforcement_learning/deep_policy.py` there is the base class which  defines dqn, double_dqn, dueling_dqn and double_dueling_dqn, sarsa, expected sarsa
2. File `reinforcement_learning/model.py` contains the torch model definitions
3. `reinforcement_learning/multi_agent_training.py`is the main traiing script.
4. `reinforcement_learning/multi_agent_training_progressive_incremental.py` and `reinforcement_learning/multi_agent_training_progressive.py` contain the training code where the agent is trained with incrementally increasing environment difficulty (otherwise almost identical to `reinforcement_learning/multi_agent_training.py`)
5. `GraphObsForRailEnv.py`contains the experimental graph observation.


