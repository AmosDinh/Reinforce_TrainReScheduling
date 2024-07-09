# Reinforce_TrainReScheduling

# Ziel

We seek to minimize the time it takes to bring all the agents to their respective target.

# Usage
    git clone https://github.com/AmosDinh/Reinforce_TrainReScheduling
    cd Reinfore_TrainReScheduling

    conda create --n rl python=3.8

    conda activate rl

    pip install -r requirements.txt

    cd src 

    python execute_local_sweep.py sweeps/sweep_policies.yaml


    

# (Not required) Weights and Biases usage:
Set wandb key on linux: WANDB_API_KEY=YOUR_API_KEY 
Our tracked metrics: [https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling](https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling)
<br>


# Files:
1. In file `reinforcement_learning/deep_policy.py` there is the base class which  defines dqn, double_dqn, dueling_dqn and double_dueling_dqn, sarsa, expected sarsa
2. File `reinforcement_learning/model.py` contains the torch model definitions
3. `reinforcement_learning/multi_agent_training_progressive_incremental.py` and `reinforcement_learning/multi_agent_training_progressive.py` contain the training code where the agent is trained with incrementally increasing environment difficulty (otherwise almost identical to `reinforcement_learning/multi_agent_training.py`)
4. 


