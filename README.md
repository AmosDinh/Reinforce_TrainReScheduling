# Reinforce_TrainReScheduling

# Ziel

We seek to minimize the time it takes to bring all the agents to their respective target.

Freeze conda env to .yaml

    env export --no-builds > environment.yml

Create conda env from .yaml

    conda env create -f environment.yml

Update conda env with .yaml

    conda env update --file environment.yml --prune
