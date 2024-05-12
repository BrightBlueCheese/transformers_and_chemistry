# import utils
from ..utils import fn_load_tokenizer_chembart
import torch
import subprocess
import os
import importlib
from transformers import RobertaTokenizer


# Global parameters
MAX_SEQ_LENGTH = 512
# tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = fn_load_tokenizer_chembart(max_seq_length=MAX_SEQ_LENGTH)

# Model hyperparameters
# VOCAB_SIZE = tokenizer.vocab_size # Chemberta Tokenizer added some tokens and "vocab_size" is a fixed attribute
VOCAB_SIZE = len(tokenizer)
PAD_TOKEN_ID = tokenizer.pad_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id
# NUM_STEP on the train_chembart side
ENCODER_FFN_DIM = 624
DECODER_FFN_DIM = 624
D_MODEL = 624

# < SINCE Bart is Encoder-Decoder Transformers 6//2 == 3 >
ENCODER_ATTENTION_HEADS = 2 ########
DECODER_ATTENTION_HEADS = 2 #########
ENCODER_LAYERS = 2 #####
DECODER_LAYERS = 2 #######

NUM_LABELS = 105
DROPOUT = 0.1
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
NUM_DATASET_SIZE = 30  ##################
NUM_HPC_PROCESS = 10 # process for csv reading

NUM_DEVICE = torch.cuda.device_count()   # this is for num_workers = 4 * 3 = 12
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

NAME_MODEL = "ChemBart_Small_30m" #########
DIR_CHECKPOINT = f"{dir_current}/saved_models/{NAME_MODEL}"
DIR_LOG = f"{dir_current}/saved_models/{NAME_MODEL}/tb_logs"

