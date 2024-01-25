# ELG (Ensemble of Local and Global policies)
This repository is the code of https://arxiv.org/abs/2308.14104, which ensembles a transferrable local policy to boost generalization. 
Our code is built on the code of POMO[1]. We provide the trained models to reproduce the test results in the paper.  

## Test ELG* on VRPLIB[2, 4]

Under the ELG/CVRP folder, use the default settings in *config.yml*, run

```
python test_vrplib.py
```
You can choose the *vrplib_set* config from *{X, XXL}* to test on two different VRPLIB sets. 
## Train ELG* on CVRP

First, generate the validation sets by

```
python generate_data.py
```

Modify the *load_checkpoint* config in *config.yml* to Null (i.e., *load_checkpoint*: ), and run

```
python train.py
```

## Test ELG* on TSPLIB[4]

Under the ELG/TSP folder, use the default settings in *config.yml*, and run

```  
python test_tsplib.py
```

## Train ELG* on TSP

First, generate the validation sets by

```
python generate_data.py
```

Modify the *load_checkpoint* term in *config.yml* to Null (i.e., *load_checkpoint*: ), and run

```
python train.py
```



Reference:

[1] Kwon, Y.-D.; Choo, J.; Kim, B.; Yoon, I.; Gwon, Y.; and Min, S. 2020. POMO: Policy optimization with multiple optima for reinforcement learning. In *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 21188–21198. Virtual.

[2] Uchoa, E.; Pecin, D.; Pessoa, A.; Poggi, M.; Vidal, T.; and Subramanian, A. 2017. New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3): 845–858.

[3] Reinelt, G. 1991. TSPLIB - A traveling salesman problem library. ORSA Journal on Computing, 3(4): 376–384.

[4] Arnold, F.; Gendreau, M.; and S¨orensen, K. 2019. Efficiently solving very large-scale routing problems. Computers & Operations Research, 107: 32–42.
