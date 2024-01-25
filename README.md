# ELG (Ensemble of Local and Global policies)

This repository is the code of the https://arxiv.org/abs/2308.14104, which ensembles a transferrable local policy to boost generalization. We provide the trained models to reproduce the test results in the paper.  

## Test ELG-POMO on VRPLIB

Under the ELG/CVRP folder, use the default settings in *config.yml*, run

```bash
python test_vrplib.py
```

You can choose the *vrplib_set* config from *{X, XXL}* to test on two different VRPLIB sets. 

## Test ELG-POMO on TSPLIB

Under the ELG/TSP folder, use the default settings in *config.yml*, and run

```bash
python test_tsplib.py
```

## Train ELG-POMO on CVRP or TSP

First, generate the validation sets by

```bash
python generate_data.py
```

Modify the *load_checkpoint* config in *config.yml* to Null (i.e., *load_checkpoint*: ), and run

```bash
python train.py
```
