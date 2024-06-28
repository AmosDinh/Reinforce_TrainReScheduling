you can use the sweep yaml together with wandb
:
in console:
<<< wandb sweep sweep.yaml


>>>
wandb: Creating sweep from: sweep.yaml
wandb: Creating sweep with ID: m8tthb4u
wandb: View sweep at: https://wandb.ai/Reinforce_Team/Reinforce_TrainReScheduling-reinforcement_learning/sweeps/m8tthb4u
wandb: Run sweep agent with: wandb agent Reinforce_Team/Reinforce_TrainReScheduling-reinforcement_learning/m8tthb4u

Sweep kann parallelisiert werden: Einfach skript mehrmals ausführen zb verschiedene Terminal Windows: Skript holt sich vom WandB server die Hyperparameterkonfigurationen, welche noch nicht benutzt wurden
