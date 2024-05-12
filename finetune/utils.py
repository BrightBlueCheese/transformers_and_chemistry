from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import numpy as np
import os

def model_evalulator(array_predictions, 
                     array_labels, 
                     dataset_dict:dict, 
                     # device:str, 
                     show_plot:bool=True, 
                     print_result:bool=True):
    
    if print_result:        
        print(f"Dataset : {dataset_dict['dataset_name']}")
        print("N:", array_labels.shape[0])
    
    fig, ax = plt.subplots()

    if dataset_dict['dataset_type'] == 'classification':
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(array_labels, array_predictions)
        # Calculate AUC
        # roc_auc = auc(fpr, tpr)
        metric = roc_auc_score(array_labels, array_predictions) # AUC
        metric2 = None
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {metric:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim(0, 1)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        if print_result:
            print(f"ROC-AUC score : {metric} - higher the better")

    else:
        metric = mean_squared_error(array_labels, array_predictions, squared=False) #RMSE
        r2 = r2_score(array_labels, array_predictions)
        metric2 = mean_absolute_error(array_labels, array_predictions) # MAE
        ax.scatter(array_labels, array_predictions)
        ax.set_title("Scatter Plot of Labels vs Predictions")
        ax.set_xlabel("Labels")
        ax.set_ylabel("Predictions")

        if print_result:
            print("R2:", r2)
            print("Root Mean Square Error:", metric)
            print("Mean Absolute Error:", metric2)

    correlation, p_value = spearmanr(array_labels, array_predictions)

    if print_result:
        print("Spearman correlation:", correlation)
        print("p-value:", p_value)
        print("=======================================")
    
    xmin, xmax = ax.get_xlim()
    ax.set_ylim(xmin, xmax)
    
    if not show_plot:
        plt.ioff()
        plt.clf()
        plt.close()
    else :
        plt.show()
    
    # metrict 1 - ROC score (classification) | RMSE (regression)
    # metric 2 - None (classification) | MAE ( regression)
    round_decimal = 6
    if metric2 != None:
        metric2 = round(metric2, round_decimal)

    list_p_value = str(p_value).split('e')
    p_value_mantissa = round(float(list_p_value[0]), round_decimal)
    if len(list_p_value) == 2:
        p_value_exponent = int(list_p_value[1])
    else:
        p_value_exponent = None
    
    return [dataset_dict['dataset_name'], 
            dataset_dict['dataset_type'], 
            round(metric, round_decimal), 
            metric2,
            p_value_mantissa,
            p_value_exponent]

from .model_finetune import CustomFinetuneModel
import torch
def load_model_ft_with_epoch(class_model_ft,
                             target_epoch:int,
                             dir_model_ft:str,
                             name_model_ft:str):
    # dir_model_ft level 1
    # ex /main/model_mtr/model_mtr_ep/dataset
    
    dir_all_model_ft = f"{dir_model_ft}/{name_model_ft}/version_0/checkpoints/"
    list_files_in_dir_model_ft = os.listdir(dir_all_model_ft)
    # extension = '.ckpt'
    extension = '.pt'
    list_model_ft_in_the_dir = sorted(list_files_in_dir_model_ft, key=lambda x: float(x.split('=')[-1].split('.')[0]))
    
    print(f"Loaded model with epoch {target_epoch}")
    dir_target_model_ft = f"{dir_all_model_ft}/{list_model_ft_in_the_dir[target_epoch]}"
    
    # class_model_ft.load_from_checkpoint(dir_target_model_ft)
    
    loaded_state_dict = torch.load(dir_target_model_ft)
    class_model_ft.load_state_dict(loaded_state_dict['state_dict'])
    
    return class_model_ft # now is model_ft

from scipy.stats import rankdata
# rankdata does not consider decimal places!
def rank_value(list_value, 
               dataset_dict:dict, 
               is_loss:bool=True):
    
    list_value = np.array(list_value)
    # since epoch starts from 0.
    if dataset_dict['dataset_type'] == 'classification' and not is_loss:
        return np.array(rankdata(list_value * -100000, method='min')) - 1
        
    elif dataset_dict['dataset_type'] == 'regression' or (dataset_dict['dataset_type'] == 'classification' and is_loss):
        return np.array(rankdata(list_value * 100000, method='min')) - 1

def model_ft_namer(dataset_name):
    # return f"ChemLlama_ft_{dataset_name}"
    return f"{dataset_name}"