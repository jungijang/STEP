# STEP: Scalable Higher-Order Interaction Prediction on Trillion-scale Tensors via Entity Pruning

This repository is the implementation for "STEP: Scalable Higher-Order Interaction Prediction on Trillion-scale Tensors via Entity Pruning", submitted to KDD 2026 (August Cycle).

## Code Information
All codes are implemented by PyTorch.
This repository contains the code for STEP.

* We provide the following codes for STEP. 
  * `model.py`: STEP + Tensor factorization model is defined. There are six original tensor factorization methods.
    * CP: CP decomposition.
    * SCP: CP decomposition combined with STEP.
    * Tucker: Tucker decomposition.
    * STucker: Tucker decomposition combined with STEP.
    * CostCo: CostCo.
    * SCostCo: CostCo combined with STEP.
    * MDMTF: $M^2DMTF$.
    * SMDMTF: $M^2DMTF$ combined with STEP.
  * `topk_predict_base.py`: base code for top-k prediction.
  * `topk_predict.py`: GPU friendly code for top-k prediction.
  * `topk_predictN.py`: GPU friendly code for top-k prediction for higher-order tensors.
  * `main_sg_base.py`: the demo code for CP and Tucker decomposition on SG dataset.
  * `main_sg_mdmtf.py`: the demo code for $M^2DMTF$ on SG dataset.  
  * `main_sg_neat.py`: the demo code for NeAT on SG dataset.  
  * `main_sg_step_w_multilinear.py`: the demo code for SCP and STucker on SG dataset.  
  * `main_sg_step_w_nonlinear.py`: the demo code for SCostCo and SMLP on SG dataset.
  * `main_sg_smdmtf.py`: the demo code for SMDMTF on SG dataset.
  * `main_sg_sneat.py`: the demo code for SNeat on SG dataset.

## Requirements

Before running demos, you need to install the requirements:

```
pip install -r requirements.txt
```

## Demo

Currently, we provide the demo codes for SG dataset.
You can change a model by modifying the code in the lines where the model is defined in the `main` codes.
If you are interested in the Gowalla or Yahoo dataset, please modify the code accordingly.
Due to space limitations, we will upload the DDS data separately.
To run a demo of our proposed model, you run the code with the following command:
* run the code for base models (e.g., CP and Tucker decomposition) on SG dataset.
```
python main_sg_base.py
```

* run the code for STEP + CP or Tucker on SG dataset.
```
python main_sg_step_w_multilinear.py
```

* run the code for STEP + CostCo or MLP on SG dataset.
```
python main_sg_step_w_nonlinear.py
```

* run the code for $M^2DMTF$ on SG dataset.
```
python main_sg_mdmtf.py
```

* run the code for STEP + $M^2DMTF$ on SG dataset.
```
python main_sg_mdmtf.py
```

* run the code for NeAT on SG dataset.
```
python main_sg_neat.py
```

* run the code for STEP + NeAT on SG dataset.
```
python main_sg_sneat.py
```
