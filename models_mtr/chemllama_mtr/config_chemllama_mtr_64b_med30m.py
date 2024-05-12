from ..utils import fn_load_tokenizer_chemllama
import torch
import subprocess
import os
import importlib

# Global parameters
MAX_SEQ_LENGTH = 512
tokenizer = fn_load_tokenizer_chemllama(max_seq_length=MAX_SEQ_LENGTH)

# Model hyperparameters
# VOCAB_SIZE = tokenizer.vocab_size
VOCAB_SIZE = len(tokenizer)
PAD_TOKEN_ID = tokenizer.pad_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 768
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN_LAYERS = 7
NUM_LABELS = 105
ATTENTION_DROPOUT = 0.0
LEARNING_RATE = 0.0001

# Dataset
KEY_DESCRIPTOR_LIST = "105_descriptors"
DIR_DESCRIPTOR_LIST = "./descriptor_list.json"

pwd_py = subprocess.run(["pwd"], stdout=subprocess.PIPE)
dir_current = pwd_py.stdout.decode("utf-8").strip()
# dir_master = "".join(["/"+x for x in dir_current.split('/')[1:-1]])

NAME_DATASET = "30m_chunk_property_znormed"
DIR_DATASET = f"{dir_current}/dataset_mtr/dataset/{NAME_DATASET}"
NUM_DATASET_SIZE = 30
NUM_HPC_PROCESS = 10 # process for csv reading

NUM_DEVICE = torch.cuda.device_count() # num_workers = 4 * 3 = 12
NUM_WORKERS_MULTIPLIER = 3
TRAIN_SIZE_RATIO = 0.8
BATCH_SIZE_TRAIN = 64
# BATCH_SIZE_VALID = int(BATCH_SIZE_TRAIN * 1.5)
BATCH_SIZE_VALID = 64


# Compute Related
ACCELERATOR = "gpu"
DEVICES = [x for x in range(NUM_DEVICE)]  # [0, 1, 2, 3]
MIN_EPOCHS = 1
MAX_EPOCHS = 7
PRECISION = 32

NAME_MODEL = "ChemLlama_Medium_30m"
DIR_CHECKPOINT = f"{dir_current}/saved_models/{NAME_MODEL}"
DIR_LOG = f"{dir_current}/saved_models/{NAME_MODEL}/tb_logs"
