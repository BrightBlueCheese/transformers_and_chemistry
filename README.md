# The Role of Model Architecture and Scale in Predicting Molecular Properties: Insights from Fine-Tuning RoBERTa, BART, and LLaMA

This repository is the official implementation of [The Role of Model Architecture and Scale in Predicting Molecular Properties: Insights from Fine-Tuning RoBERTa, BART, and LLaMA](https://arxiv.org/abs/2405.00949). 


## Requirements

To install requirements:

```setup
pip install -r requirements_cuda118.txt
```

>📋  The experiments were done under CUDA 11.8

## Dataset
1. Open ```stage_1_create_canonical_smiles.ipynb``` and run the cells
(** You can skip this part if you want to use the csv file from github with "canonical_smiles_xxx.csv")
2. ```python stage_2_descriptors_for.py```
3. Open ```stage_3_preprocess_smiles_property.ipynb``` and run the cells.

>📋  check `instruction_dataset_mtr.txt`

## Training

To train the model(s) in the paper, move to `transformers_and_chemistry` (the main directory) and run:

For Multitask Regression (MTR) model training
```train-mtr
python -m models_mtr.train_chemXXX
```

For Finetune model training
```train-ft
python -m finetune.run_auto_XXX_bulk
```

>📋  check `instruction_models_mtr.txt` and `instruction_finetune.txt`

## Evaluation

To evaluate the Finetune models, move to `transformers_and_chemistry/eval` (the main directory) and:

1. Open ```Eval_Tabular.ipynb``` and run the cells


## Pre-trained Models

Since this experiments yields multiple MTR and Finetune models, downloading pre-trained model is not available. But you can reproduce them through this code.

## Contributing

>📋  MIT
