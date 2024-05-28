# Reinforce_TrainReScheduling

Freeze conda env to .yaml

    conda env export --name rl > environment.yml

Create conda env from .yaml

    conda env create -f environment.yml

Update conda env with .yaml

    conda env update --file environment.yml --prune
