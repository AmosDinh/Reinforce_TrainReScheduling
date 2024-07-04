# Reinforce_TrainReScheduling

# Ziel

We seek to minimize the time it takes to bring all the agents to their respective target.

# Usage

    conda create --name rl python=3.10.14

    pip install -r requirements.txt

# Weights and Biases usage:
Key auf Whatsapp (wird beim ersten ausführen nachgefragt, ansonsten auf linux: export WANDB_API_KEY=YOUR_API_KEY) <br>
Loggt einfach die Tensorboard runs (benutzt den tensorboard syntax mit: writer.add_scalar(...)) <br>
Für einen run kann der Run-Name angepasst werden in multi_agent_training.py: <br>
`wandb.init(sync_tensorboard=True, name="flatland-rl_run1", project='Reinforce_TrainRescheduling')`

<br>
Link: [https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling](https://wandb.ai/Reinforce_Team/Reinforce_TrainRescheduling)


# Agents:
In file `reinforcement_learning/deep_policy` there is the base class which  defines dqn, double_dqn, dueling_dqn and double_dueling_dqn

