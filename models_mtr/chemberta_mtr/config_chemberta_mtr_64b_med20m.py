# import utils
from ..utils import fn_load_tokenizer_chemberta
from ..utils import fn_load_tokenizer_chemllama
import torch
import subprocess
import os
import importlib
from transformers import RobertaTokenizer


# Global parameters
# MAX_SEQ_LENGTH = 202
MAX_SEQ_LENGTH = 512
# tokenizer = RobertaTokenizer.from_pretrained(
#         "DeepChem/ChemBERTa-77M-MTR",
#         model_max_length=MAX_SEQ_LENGTH, 
#         padding_side="right",
#     )
# tokenizer = fn_load_tokenizer_chemllama(max_seq_length=MAX_SEQ_LENGTH)
tokenizer = fn_load_tokenizer_chemberta(max_seq_length=MAX_SEQ_LENGTH) # Embed index error!!!

# Model hyperparameters
# VOCAB_SIZE = tokenizer.vocab_size # Chemberta Tokenizer added some tokens and "vocab_size" is a fixed attribute
VOCAB_SIZE = len(tokenizer)
# VOCAB_SIZE = 600 if len(tokenizer) <= 600 else len(tokenizer) 
PAD_TOKEN_ID = tokenizer.pad_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id
# BOS_TOKEN_ID = tokenizer.cls_token_id
# EOS_TOKEN_ID = tokenizer.sep_token_id
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 768
NUM_ATTENTION_HEADS = 8 ########
NUM_HIDDEN_LAYERS = 8   ########
NUM_LABELS = 105
ATTENTION_PROBS_DROPOUT_PROB = 0.1
HIDDEN_DROPOUT_PROB = 0.1
LEARNING_RATE = 0.0001

# Dataset
KEY_DESCRIPTOR_LIST = "105_descriptors"
DIR_DESCRIPTOR_LIST = "./descriptor_list.json"

pwd_py = subprocess.run(["pwd"], stdout=subprocess.PIPE)
dir_current = pwd_py.stdout.decode("utf-8").strip()
# dir_master = "".join(["/"+x for x in dir_current.split('/')[1:-1]])

NAME_DATASET = "30m_chunk_property_znormed"
DIR_DATASET = f"{dir_current}/dataset_mtr/dataset/{NAME_DATASET}"
NUM_DATASET_SIZE = 20  ##################
NUM_HPC_PROCESS = 10 # process for csv reading


NUM_DEVICE = torch.cuda.device_count()     # num_workers = 4 * 3 = 12
# NUM_DEVICE = 1
NUM_WORKERS_MULTIPLIER = 3
TRAIN_SIZE_RATIO = 0.8
BATCH_SIZE_TRAIN = 64
# BATCH_SIZE_VALID = int(BATCH_SIZE_TRAIN * 1.5)
BATCH_SIZE_VALID = 64

# Compute Related
ACCELERATOR = "gpu"
# ACCELERATOR = "cpu"
DEVICES = [x for x in range(NUM_DEVICE)]  # [0, 1, 2, 3]
MIN_EPOCHS = 1
MAX_EPOCHS = 7
PRECISION = 32
# PRECISION = 16

NAME_MODEL = "ChemBerta_Medium_20m" #########
DIR_CHECKPOINT = f"{dir_current}/saved_models/{NAME_MODEL}"
DIR_LOG = f"{dir_current}/saved_models/{NAME_MODEL}/tb_logs"
