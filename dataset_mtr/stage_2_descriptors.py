import pandas as pd
import numpy as np
import os
import utils
import multiprocessing
import json
import warnings

# Ignore the specific RuntimeWarning
warnings.filterwarnings("ignore", message="overflow encountered in cast")


random_seed = 1004
np.random.seed(random_seed)

from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

cur_dir = os.getcwd()
name_dataset_dir = "dataset"

name_csv_to_save = "30m_chunk_property"
dir_output = f"{cur_dir}/dataset/{name_csv_to_save}"
os.makedirs(dir_output, exist_ok=True)

# You need to run this file with `# FIRST` first while comment the lines with `# SECOND`
# Then comment `# First` and uncomment `# SECOND` and run again.
# You may change the `num_process` according to your RAM and Process status

############################################
num_first_chunk = 1 # FIRST
# num_first_chunk = 31  # SECOND

for num_csv_chunk in range(num_first_chunk, num_first_chunk+30): # FIRST
# for num_csv_chunk in range(num_first_chunk, num_first_chunk + 3):  # SECOND

    name_csv = f"canonical_smiles_{str(num_csv_chunk).zfill(3)}.csv"
    df_smile_chunk = pd.read_csv(f"{cur_dir}/{name_dataset_dir}/{name_csv}")

    ####### DF for molcules #####
    num_processes = 10
    list_for_HPC_molcule = utils.fn_split_dataframe_row(df_smile_chunk, num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    # Use multiprocessing to calculate properties for each SMILES
    list_smiles_description = pool.map_async(
        utils.fn_smile_description, list_for_HPC_molcule
    )

    pool.close()
    pool.join()

    ###########################

    df_smile_chunk = pd.concat(list_smiles_description.get())
    df_smile_chunk = df_smile_chunk.reset_index(drop=True)

    #########################

    list_for_HPC_property = utils.fn_split_dataframe_row(df_smile_chunk, num_processes)
    #########################
    key_descriptor_list = "105_descriptors"

    list_cooked_property = list()
    for local_df_idx, local_df in enumerate(list_for_HPC_property):
        print(f"stage : {local_df_idx}")
        pool = multiprocessing.Pool(processes=64)
        list_local_property = pool.starmap_async(
            utils.fn_calculate_properties_for,
            zip(local_df["molcule"], [key_descriptor_list] * local_df.shape[0]),
        )
        pool.close()
        pool.join()
        list_local_property = np.array(list_local_property.get())
        print(list_local_property.shape)
        list_cooked_property.append(list_local_property)

    array_cooked_property = np.vstack(list_cooked_property)

    with open("../descriptor_list.json", "r") as js:
        list_descriptor = json.load(js)[key_descriptor_list]

    df_cooked_property = pd.DataFrame(array_cooked_property, columns=list_descriptor)
    df_cooked_property.insert(loc=0, column="SMILES", value=df_smile_chunk["SMILES"])

    df_cooked_property.to_csv(
        f"{dir_output}/{name_csv_to_save}_{str(num_csv_chunk).zfill(3)}.csv",
        index=False,
    )

    print(f"Chunk No.{num_csv_chunk} Done.")
