import sys
import os
# Select GPU if you have multiple ones.
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import torch
import numpy as np
import pandas as pd
import warnings
import lightning as L
torch.set_float32_matmul_precision('high')

# Filter out FutureWarning and UnderReviewWarning messages from pl_bolts
warnings.filterwarnings("ignore", module="pl_bolts")

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from models_mtr
from models_mtr import utils
# from models_mtr.chemberta_mtr.chemberta_mtr import ChemBerta
from models_mtr.chemberta_mtr import config_chemberta_mtr_64b_sm10m as config


from . import molnet
from .model_finetune import CustomFinetuneModel
from .datamodule_finetune import CustomFinetuneDataModule
from . import auto_evaluator

print(os.path.dirname(__file__))


torch.manual_seed(1004)
np.random.seed(1004)

print(os.getcwd())
for ver_ft in range(0, 5):
    dir_main = "./saved_models"
    name_model_mtr = "ChemBerta_Medium_10m" #####
    model_class = "roberta"                #####
    model_version = 0
    tokenizer = utils.fn_load_tokenizer_chemberta(
        max_seq_length=config.MAX_SEQ_LENGTH,
    )
    max_length = config.MAX_SEQ_LENGTH
    molnet_dict = molnet.molnet_dict
    list_dataset_to_finetune = list(molnet_dict.keys())
    num_workers = config.NUM_WORKERS_MULTIPLIER
    dir_model_mtr_to_save = f"./save_models_finetune/ft_version_{ver_ft}"
    name_model_mtr_to_save = f"{name_model_mtr}"
    batch_size_pair = [64, 64] # [train, valid(test)]
    lr_pair = [0.01, 0.01] # [lr_regression, lr_classification] # this is peak lr
    overwrite_level_2 = True
    overwrite_level_3 = True
    epochs = 7
    use_freeze = True
    
    array_level_3 = auto_evaluator.auto_evaluator_level_3(
        dir_main=dir_main,
        name_model_mtr=name_model_mtr,
        model_class=model_class,
        model_version=model_version,
        tokenizer=tokenizer,
        max_length=max_length,
        molnet_dict=molnet_dict,
        list_dataset_to_finetune=list_dataset_to_finetune,
        num_workers=num_workers,
        dir_model_mtr_to_save=dir_model_mtr_to_save,
        name_model_mtr_to_save=name_model_mtr_to_save,
        batch_size_pair=batch_size_pair,
        lr_pair=lr_pair,
        overwrite_level_2=overwrite_level_2,
        overwrite_level_3=overwrite_level_3,
        epochs=epochs,
        use_freeze=use_freeze,
    )
    
    print("============================")
    print(array_level_3.shape)
    # print(array_level_3)
    
    list_column_names_level_3 = ['model_mtr_name',
                                 'model_mtr_ep',
                                 'dataset_name', 
                                 'dataset_type', 
                                 'metric_1', 
                                 'metric_2', 
                                 'p_value_mantissa', 
                                 'p_value_exponent', 
                                 'epoch', 
                                 'loss',
                                 'loss_ranking',
                                 'metric_1_ranking']
    
    df_evaluation_level_3 = pd.DataFrame(array_level_3, columns=list_column_names_level_3)
    
    os.makedirs(f'{os.path.dirname(__file__)}/evaluations/ft_version_{ver_ft}', exist_ok=True)
    df_evaluation_level_3.to_csv(f'{os.path.dirname(__file__)}/evaluations/ft_version_{ver_ft}/{name_model_mtr}.csv', index=False)
