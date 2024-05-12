import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(1004)

from rdkit import Chem
from rdkit.Chem import Descriptors


def fn_split_dataframe_row(target_df: pd.DataFrame, target_num_core: int):
    num_row_total = target_df.shape[0]

    num_split_step = num_row_total // target_num_core
    is_remainder = 0

    if num_row_total % target_num_core != 0:
        is_remainder = -1

    list_splited_df = list()
    for idx, i in enumerate(range(0, target_num_core * num_split_step, num_split_step)):

        if idx <= ((num_row_total // num_split_step) - 1 + is_remainder):
            list_splited_df.append(target_df.iloc[i : i + num_split_step, :])

        else:
            list_splited_df.append(target_df.iloc[i:, :])

    return list_splited_df


def fn_smile_description(target_df: pd.DataFrame):
    target_df["molcule"] = target_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

    return target_df.dropna()


def fn_split_series_row(target_series: pd.Series, target_num_core: int):
    num_row_total = target_series.shape[0]

    num_split_step = num_row_total // target_num_core
    is_remainder = 0

    if num_row_total % target_num_core != 0:
        is_remainder = -1

    list_splited_df = list()
    for idx, i in enumerate(range(0, target_num_core * num_split_step, num_split_step)):
        if idx <= ((num_row_total // num_split_step) - 1 + is_remainder):
            list_splited_df.append(target_series[i : i + num_split_step])
        else:
            list_splited_df.append(target_series[i:])

    return list_splited_df


#################### NEW #######################
import json
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Descriptors


def fn_calculate_properties_for(target_molcule, target_descriptor_list_key):
    # print(target_molcule.shape)
    # sorry about this messyness XD
    try:
        with open("./descriptor_list.json", "r") as js:
            list_descriptor = json.load(js)[target_descriptor_list_key]
    except:
        with open("../descriptor_list.json", "r") as js:
            list_descriptor = json.load(js)[target_descriptor_list_key]

    calculator = MolecularDescriptorCalculator(list_descriptor)
    # list_property = list()
    # for m in [target_molcule]:
    #     local_property = calculator.CalcDescriptors(m)
    #     list_property.append(local_property)
    local_property = calculator.CalcDescriptors(target_molcule)
    local_property = np.array(local_property, dtype="float32")
    # array_smile_and_property = np.hstack((target_smile, local_property))
    # now is an array but won't assign a new var name due to memory optimization
    # list_property = np.array(list_property)
    # array_smile_and_property = np.hstack((target_smile, list_property))
    array_smile_and_property = np.nan_to_num(
        local_property, nan=0.0, posinf=0.0, neginf=0.0
    )

    return array_smile_and_property



# class class_calculate_properties_vec:
#     def __init__(self, target_df):
#         self.target_df = target_df

#     def fn_calculate_properties_vec(self, target_descriptor): # single descriptor as a str
#         calculator = MolecularDescriptorCalculator(target_descriptor)
#         array_property = self.target_df['molcule'].apply(lambda x: calculator.CalcDescriptors(x))
#         array_property = np.nan_to_num(array_property,
#                                        nan=0.0,
#                                        posinf=0.0,
#                                        neginf=0.0)
#         return array_property


def fn_save_df_HPC_sub(
    target_df, target_df_index, target_dir_to_save, target_name_to_save
):

    target_df.to_csv(
        f"{target_dir_to_save}/{target_name_to_save}_{str(target_df_index).zfill(3)}.csv",
        index=False,
    )


def fn_create_sub_df_for_HPC(target_num_row_per_chunk, target_df_base):

    num_row_total = target_df_base.shape[0]
    num_chunk = num_row_total // target_num_row_per_chunk

    list_df_for_HPC = list()
    for i, chunk in enumerate(range(0, num_row_total, target_num_row_per_chunk)):
        if i + 1 != num_chunk:
            local_df = target_df_base.iloc[chunk : chunk + target_num_row_per_chunk, :]
            list_df_for_HPC.append(local_df)
        else:
            local_df = target_df_base.iloc[chunk:, :]
            list_df_for_HPC.append(local_df)
            break
    return list_df_for_HPC, np.arange(1, len(list_df_for_HPC) + 1).tolist()


import multiprocessing
import os


def fn_save_df_HPC(
    target_num_row_per_chunk,
    target_num_process,
    target_df_base,
    target_dir_to_save,
    target_name_to_save,
    overwrite: bool = False,
):

    print("HPC working on it.")
    try:
        os.makedirs(target_dir_to_save, exist_ok=overwrite)
    except:
        return print(
            "This directory is already existed. And you sat overwrite=False. Check it again."
        )

    list_df_for_HPC, list_indices_for_HPC = fn_create_sub_df_for_HPC(
        target_num_row_per_chunk=target_num_row_per_chunk, target_df_base=target_df_base
    )
    num_chunk = len(list_indices_for_HPC)

    pool = multiprocessing.Pool(processes=target_num_process)
    pool.starmap_async(
        fn_save_df_HPC_sub,
        zip(
            list_df_for_HPC,
            list_indices_for_HPC,
            [target_dir_to_save] * num_chunk,
            [target_name_to_save] * num_chunk,
        ),
    )
    pool.close()
    pool.join()
    print("HPC done")


class class_read_csv:
    def __init__(self, target_list_descriptor):

        self.target_list_descriptor = target_list_descriptor

        dict_col_dtype = {"SMILES": "str"}
        for col in target_list_descriptor:
            dict_col_dtype[col] = "float32"
        self.dict_col_dtype = dict_col_dtype

    def fn_read_csv(self, target_dir_csv):

        return pd.read_csv(target_dir_csv, dtype=self.dict_col_dtype)

    def fn_read_csv_HPC(self, target_list_csv_dir: str, target_num_process: int):

        pool = multiprocessing.Pool(processes=target_num_process)
        list_df = pool.map_async(self.fn_read_csv, target_list_csv_dir)
        pool.close()
        pool.join()

        df_final_csv = pd.concat(list_df.get())
        df_final_csv = df_final_csv.reset_index(drop=True)

        name_column_to_drop = "Unnamed: 0"
        if name_column_to_drop in df_final_csv.columns:
            df_final_csv = df_final_csv.drop(columns=name_column_to_drop)

        return df_final_csv


def fn_zscore_normalizer_col(target_series):
    local_mean = target_series.values.mean()
    local_std = target_series.values.std()

    target_series = (target_series - local_mean) / local_std

    return target_series


def fn_zscore_normalizer_HPC(target_df, target_num_process: int):

    list_series_for_norm = list()
    for local_col in target_df.columns[1:]:
        list_series_for_norm.append(target_df[local_col])

    pool = multiprocessing.Pool(processes=target_num_process)
    list_series_normed = pool.map_async(fn_zscore_normalizer_col, list_series_for_norm)
    pool.close()
    pool.join()

    dict_series_normed = {"SMILES": target_df["SMILES"]}
    for local_ser in list_series_normed.get():
        dict_series_normed[local_ser.name] = local_ser.values

    df_normed = pd.DataFrame(dict_series_normed)

    return df_normed
