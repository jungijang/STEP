# SCOUT: Coupling-Free Bounds for Trillion-Scale Top-k Retrieval in Sparse Tensor Factorization

This repository is the implementation for "SCOUT: Coupling-Free Bounds for Trillion-Scale Top-k Retrieval in Sparse Tensor Factorization", submitted to SIGMOD 2027 (Round 1).

## Code Information
All codes are implemented by PyTorch.
This repository contains the code for SCOUT.

* We provide the following codes for SCOUT. 
  * `modelN.py`: SCOUT + Tensor factorization model is defined. There are six SCOUT + TF methods.
    * SCPN: CP decomposition combined with SCOUT.
    * STuckerN: Tucker decomposition combined with SCOUT.
    * SCostCoN: CostCo combined with SCOUT.
    * SMLPN: MLP combined with SCOUT.
    * SNeAT: NeAT combined with SCOUT.
    * SMDMTF: $M^2DMTF$ combined with SCOUT.
  * `modelN_base.py`: Original tensor factorization model is defined. There are six original tensor factorization methods.
    * CP, Tucker, CostCo, NeAT, MDMTF, MLP.
  * `topk_predictN.py`: GPU friendly code for top-k prediction for higher-order tensors.
  * `nway_train.py`: the train and test code for SCOUT.
  * `nway_train_base.py`: the train and test code for original TF methods.

## Requirements

Before running demos, you need to install the requirements:
```
pip install -r requirements.txt
```

## Demo

Currently, we provide the demo codes for the origianl TF methods and SCOUT on Gowalla dataset.
If you are interested in another dataset, please modify the shell accordingly.
Due to space limitations, we have uploaded only relatively lightweight data.
To run a demo of our proposed model, you run the code with the following command:
* run the code for SCOUT.
```
bash run_scout.sh
```

* run the code for the original TF methods.
```
bash run_base.sh
```
