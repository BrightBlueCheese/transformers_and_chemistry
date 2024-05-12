import sys
import os
import re
import pandas as pd
import numpy as np

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from models_mtr
from models_mtr.chembart_mtr.chembart_mtr import ChemBart
from models_mtr.chemberta_mtr.chemberta_mtr import ChemBerta
from models_mtr.chemllama_mtr.chemllama_mtr import ChemLlama

from .datamodule_finetune import CustomFinetuneDataModule
from .model_finetune import CustomFinetuneModel
from . import utils

def auto_evaluator_level_3(
    dir_main:str,
    name_model_mtr:str,
    model_class:str,
    model_version:int,
    tokenizer,
    max_length:int,
    molnet_dict:dict,
    list_dataset_to_finetune:list,
    num_workers:int,
    dir_model_mtr_to_save:str,
    name_model_mtr_to_save:str,
    batch_size_pair=[32, 48],
    lr_pair=[0.01, 0.005],
    overwrite_level_2:bool=False,
    overwrite_level_3:bool=False,
    epochs:int=7,
    use_freeze:bool=True,
):

    assert not (os.path.exists(dir_model_mtr_to_save) and overwrite_level_3 == False), f"You sat 'overwrite' False and '{dir_model_mtr_to_save}' already exists. Check it again."
    assert  model_class in ['llama', 'roberta', 'bart'], "'model_class' can only be those following : ['llama', 'roberta', 'bart']"

    # dir_main = '/scratch/ylee/ChemLLama/saved_models'
    # name_model_mtr = 'ChemLlama_Small_10m'
    dir_all_model_mtr = f"{dir_main}/{name_model_mtr}/tb_logs/{name_model_mtr}/version_{model_version}/checkpoints/"
    list_files_in_dir_model_mtr = os.listdir(dir_all_model_mtr)
    # extension = '.ckpt'
    list_model_mtr_in_the_dir = sorted(list_files_in_dir_model_mtr, key=lambda x: float(x.split('=')[-1].split('.')[0]))
    
    list_local_mtr_result = list()
    for local_model_mtr_ep, local_model_mtr in enumerate(list_model_mtr_in_the_dir):
        
        dir_model_mtr = f"{dir_all_model_mtr}/{local_model_mtr}"
        if model_class == 'bart':
            model_mtr = ChemBart.load_from_checkpoint(dir_model_mtr)
        elif model_class == 'roberta':
            model_mtr = ChemBerta.load_from_checkpoint(dir_model_mtr)
        elif model_class == 'llama':
            model_mtr = ChemLlama.load_from_checkpoint(dir_model_mtr)
        
        dir_model_mtr_ep_to_save = f"{dir_model_mtr_to_save}/{name_model_mtr_to_save}/{name_model_mtr_to_save}_{str(local_model_mtr_ep).zfill(2)}"
        array_level_2 = auto_evaluator_level_2(
            model_mtr=model_mtr,
            dir_model_mtr_ep_to_save=dir_model_mtr_ep_to_save,
            tokenizer=tokenizer,
            max_length=max_length,
            molnet_dict=molnet_dict,
            list_dataset_to_finetune=list_dataset_to_finetune,
            num_workers=num_workers,
            batch_size_pair=batch_size_pair,
            lr_pair=lr_pair,
            overwrite_level_2=overwrite_level_2,
            epochs=epochs,
            use_freeze=use_freeze,
        )

        # concatenate the current epoch number (model_mtr) to the left for the all rows
        array_ep_to_concat = np.ones((array_level_2.shape[0], 1)) * local_model_mtr_ep
        array_level_2_with_current_ep = np.hstack((array_ep_to_concat, array_level_2))
        list_local_mtr_result.append(array_level_2_with_current_ep)

    array_level_3 = np.vstack(list_local_mtr_result)
    # concatenate the current name_model_mtr_to_save to the left for the all rows
    array_name_model_mtr_to_save_to_concat = np.full((array_level_3.shape[0], 1), name_model_mtr_to_save)
    array_level_3 = np.hstack((array_name_model_mtr_to_save_to_concat, array_level_3))
    # array_level_3 shaped (number of parameter "epochs" x len(list_dataset_to_finetune) x num_model_mtr_in_the_dir, number of columns at the bottom)
    # name_model_mtr_to_save, local_model_mtr_ep, dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent, epoch, loss, loss_ranking, metric_1_ranking
    
    return array_level_3


def auto_evaluator_level_2(
    model_mtr,
    dir_model_mtr_ep_to_save:str,
    tokenizer,
    max_length:int,
    molnet_dict:dict,
    list_dataset_to_finetune:list,
    num_workers:int,
    batch_size_pair=[32, 48],
    lr_pair=[0.01, 0.005],
    overwrite_level_2:bool=False,
    epochs:int=7,
    use_freeze:bool=True
):

    """
    Evaluate the "one" pretrained MTR model through multiple finetuning benchmarking dataset.

    Parameters:
    - dir_model_mtr_ep_to_save (str): The pretrained model for MTR with epoch.
                                       EX with 0 epoch:
                                       /master_dicrectory/pre_trained_model_MTR_name/model_MTR_with_epoch
    - batch_size_pair: The pair of the train and valid(+test) batch size (e.g. [32, 48] which is [32, int(32*1.5)])
    - lr_pair: The pair of classification and regression learning rate for the finetune model (e.g. [0.01, 0.005]).
    - overwrite_level_2 (bool): If there exists such folder that has the same "dir_model_mtr_ep_to_save", overwite it.
                                Warning! This option is only for "dir_model_mtr_ep_to_save". It's sub directory and files will be overwritten!
    """
    
    
    assert not (os.path.exists(dir_model_mtr_ep_to_save) and overwrite_level_2 == False), f"You sat 'overwrite_level_2' False and '{dir_model_mtr_ep_to_save}' already exists. Check it again."
        
    # local_dataset_to_finetune is a key of molnet_dict
    list_local_finetuned_result = list()
    for local_dataset_to_finetune in list_dataset_to_finetune:
        
        dataset_dict = molnet_dict[local_dataset_to_finetune]
        dataset_dict["dataset_name"] = local_dataset_to_finetune
        
        # dir_model_ft = f"{dir_model_mtr_ep_to_save}/{dataset_dict['dataset_name']}"
        dir_model_ft = f"{dir_model_mtr_ep_to_save}"
        name_model_ft = utils.model_ft_namer(dataset_dict['dataset_name'])

        # array_level_1, model_ft, data_loader_test
        array_level_1 = auto_evaluator_level_1(
            model_mtr=model_mtr, 
            dir_model_ft=dir_model_ft, 
            name_model_ft=name_model_ft, 
            dataset_dict=dataset_dict, 
            tokenizer=tokenizer, 
            max_length=max_length,
            num_workers=num_workers,
            batch_size_pair=batch_size_pair,
            lr_pair=lr_pair,
            epochs=epochs, 
            use_freeze=use_freeze,
        )
        
        list_local_finetuned_result.append(array_level_1)
        
    array_level_2 = np.vstack(list_local_finetuned_result)
    # array_level_2 shaped (number of epochs x len(list_dataset_to_finetune), number of columns at the bottom)
    # dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent, epoch, loss, loss_ranking, metric_1_ranking
    
    return array_level_2

def auto_evaluator_level_1(
    model_mtr, 
    dir_model_ft:str, 
    name_model_ft:str, 
    dataset_dict:dict, 
    tokenizer, 
    max_length:int,
    num_workers:int, ##
    batch_size_pair=[32, 48],
    lr_pair=[0.01, 0.005],
    epochs:int=7,
    use_freeze:bool=True,
):

    """
    Automate the entire process including preparing "one" finetuning dataset + finetuing + evalulation.
    This is a step before the level 2 evaluate automation.

    Parameters:
    - model_mtr: The pretrained model for MTR.
    - dir_model_ft (str): The directory where the model to be stored.
    - name_model_ft (str): The name of the model for finetune to be titled.
                           An example of the directory of the fintuned model with 0 epoch:
                           {dir_folder}/{name_model_ft}_ep_000
    - batch_size_pair: The pair of the train and valid(+test) batch size (e.g. [32, 48] which is [32, int(32*1.5)])
    - lr_pair: The pair of classification and regression learning rate for the finetune model (e.g. [0.01, 0.005]).
    """

    assert dataset_dict['dataset_type'] in ['classification', 'regression'], '"dataset_type" must be either "classification" or "regression!!'

    csv_logger = CSVLogger(
        save_dir=dir_model_ft,
        name=name_model_ft,
        version=0,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=name_model_ft + '_vloss_{val_loss:.3f}_ep_{epoch:02d}',
        every_n_epochs=1,
        save_top_k=-1,
        enable_version_counter=False, # keep the version == 0
        save_weights_only=True,
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    
    # Load dataset for finetune
    batch_size_for_train = batch_size_pair[0]
    batch_size_for_valid = batch_size_pair[1]

    data_module = CustomFinetuneDataModule(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        max_seq_length=max_length,
        batch_size_train=batch_size_for_train,
        batch_size_valid=batch_size_for_valid,
        # num_device=int(config.NUM_DEVICE) * config.NUM_WORKERS_MULTIPLIER,
        num_device=num_workers,
    )
    data_module.prepare_data()
    data_module.setup()
    steps_per_epoch = len(data_module.train_dataloader())

    # Load model and optimizer for finetune
    lr_for_classication = lr_pair[0]
    lr_for_regression = lr_pair[1]
    if dataset_dict['dataset_type'] == 'classification':
        learning_rate = lr_for_classication
        # learning_rate = 0.005 old | new was 0.01
    elif dataset_dict['dataset_type'] == 'regression':
        learning_rate = lr_for_regression
        
    model_ft = CustomFinetuneModel(
        model_mtr=model_mtr,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=1,
        max_epochs=epochs,
        learning_rate=learning_rate,
        dataset_dict=dataset_dict,
        use_freeze=use_freeze,
    )
    
    trainer = L.Trainer(
        default_root_dir=dir_model_ft,
        # profiler=profiler,
        logger=csv_logger,
        accelerator='auto',
        devices='auto',
        # accelerator='gpu',
        # devices=[0],
        min_epochs=1,
        max_epochs=epochs,
        precision=32,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model_ft, data_module)
    trainer.validate(model_ft, data_module)

    list_validation_loss = pd.read_csv(f"{dir_model_ft}/{name_model_ft}/version_0/metrics.csv", usecols=['val_loss'])['val_loss'].dropna().tolist()[:epochs]

    # class_model_ft = CustomFinetuneModel
    # Level 1 Automation - Evaulate the finetuned model through every epoch
    array_level_1 = auto_evaluator_level_1_sub(class_model_ft=model_ft, 
                                           list_validation_loss=list_validation_loss, 
                                           dir_model_ft=dir_model_ft, 
                                           name_model_ft=name_model_ft, 
                                           data_module=data_module,
                                           dataset_dict=dataset_dict,
                                           trainer=trainer)
    
    return array_level_1

def auto_evaluator_level_1_sub(
    class_model_ft,
    list_validation_loss,
    dir_model_ft:str,
    name_model_ft:str,
    data_module,
    dataset_dict:dict,
    trainer,
):

    """
    Evaluate the finetuned model by a single finetuning dataset.

    Guides for some parameters:
    - model_mtr: The pretrained model for MTR.
    - dir_model_ft (str): The directory where the model to be stored.
    - name_model_ft (str): The name of the model for finetune to be titled.
                           An example of the directory of the fintuned model with 0 epoch:
                           {dir_folder}/{name_model_ft}_ep_000
    """
    
    array_loss_ranking = utils.rank_value(list_value=list_validation_loss, 
                                          dataset_dict=dataset_dict,
                                          is_loss=True)
    # ranking : lower the better. ranking starting from 0

    print("- Epoch starts from 0")
    print("=======================================")
    
    list_level_1 = list()
    for ep in range(len(list_validation_loss)):

        local_model_ft = utils.load_model_ft_with_epoch(class_model_ft=class_model_ft, 
                                                        target_epoch=ep,
                                                        dir_model_ft=dir_model_ft,
                                                        name_model_ft=name_model_ft)
        
        result = trainer.predict(local_model_ft, data_module)
        result_pred = list()
        result_label = list()
        for bat in range(len(result)):
            result_pred.append(result[bat][0].squeeze())
            result_label.append(result[bat][1])
        
        list_local_model_ft_result = utils.model_evalulator(array_predictions=np.hstack(result_pred),
                                                            array_labels=np.hstack(result_label),
                                                            dataset_dict=dataset_dict, 
                                                            show_plot=False,
                                                            print_result=False)
        # dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent
        
        # add epoch (starting from 0) to the right 
        list_local_model_ft_result.append(ep)
        # dataset_name, task, metric1 (RMSE or ROC-AUC), metric2 (MAE or None), p_value mantissam, p_value exponent, epoch
        
        list_level_1.append(list_local_model_ft_result)
    print("=======================================")
    print("=======================================")

    # to get the metric_1 ranking
    array_level_1 = np.array(list_level_1)
    array_metric_1 = array_level_1[:, 2].astype('float32')
    array_metric_1_ranking = utils.rank_value(list_value=array_metric_1,
                                              dataset_dict=dataset_dict,
                                              is_loss=False)

    # add loss, and ranking of the loss value to the right
    # reg: lower the better, class: higher the better
    array_level_1 = np.hstack((list_level_1, 
                               np.expand_dims(list_validation_loss, axis=1), 
                               np.expand_dims(array_loss_ranking, axis=1),
                               np.expand_dims(array_metric_1_ranking, axis=1))) 
    # dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent, epoch, loss, loss_ranking, metric_1_ranking
    
    return array_level_1 
    #################################### EX #########################################
    # list_column_names = ['dataset_name', 
    #                      'dataset_type', 
    #                      'metric_1', 
    #                      'metric_2', 
    #                      'p_value_mantissa', 
    #                      'p_value_exponent', 
    #                      'epoch', 
    #                      'loss',
    #                      'loss_ranking',
    #                      'metric_1_ranking']
    # df_evaluation_level_1 = pd.DataFrame(array_level_1, columns=list_column_names)
    #################################################################################