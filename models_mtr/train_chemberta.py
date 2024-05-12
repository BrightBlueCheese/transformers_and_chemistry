import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import LearningRateMonitor ###
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from dataset_mtr.utils import class_read_csv #1
from . import utils #2

from .chemberta_mtr.chemberta_mtr import ChemBerta
from .datamodule_mtr import CustomDeepChemDataModule


# from .chemberta_mtr import config_chemberta_mtr_64b_med30m as config #g02
# from .chemberta_mtr import config_chemberta_mtr_64b_med20m as config #g05
# from .chemberta_mtr import config_chemberta_mtr_64b_sm10m as config #g06
# from .chemberta_mtr import config_chemberta_mtr_64b_med10m as config #g07
# from .chemberta_mtr import config_chemberta_mtr_64b_sm20m as config #g02
from .chemberta_mtr import config_chemberta_mtr_64b_sm30m as config #g04

from transformers import RobertaTokenizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# (chemllm) [ylee@g002 transformers_and_chemistry]$ python -m models_mtr.train_chemllama
# python -m models_mtr.train_chemllama

# speed-up option
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    print(f"Model Name: {config.NAME_MODEL} | Dataset Size : {config.NUM_DATASET_SIZE}m")
    
    # tb_logger = TensorBoardLogger(
    #     save_dir=config.DIR_LOG,
    #     # save_dir='tb_logs',
    #     name=config.NAME_MODEL
    # )
    
    csv_logger = CSVLogger(
        save_dir=config.DIR_LOG,
        name=config.NAME_MODEL
    )

    profiler = PyTorchProfiler(
        dirpath=config.DIR_LOG,
        filename=config.NAME_MODEL,
    )
    
    list_descriptor = utils.fn_load_descriptor_list(
        key_descriptor_list=config.KEY_DESCRIPTOR_LIST,
        dir_descriptor_list=config.DIR_DESCRIPTOR_LIST,
    )

    dir_csv = config.DIR_DATASET
    list_csv_name = sorted(os.listdir(dir_csv))
    list_dir_csv = list()
    for f in list_csv_name:
        list_dir_csv.append(os.path.join(dir_csv, f))

    print("== Reading Datasaset. It will take a while. ==")
    
    csv_reader = class_read_csv(target_list_descriptor=list_dir_csv)
    df = csv_reader.fn_read_csv_HPC(target_list_csv_dir=list_dir_csv[:config.NUM_DATASET_SIZE], target_num_process=config.NUM_HPC_PROCESS)

    # tokenizer = RobertaTokenizer.from_pretrained(
    #     "DeepChem/ChemBERTa-77M-MTR",
    #     model_max_length=config.MAX_SEQ_LENGTH, 
    #     padding_side="right",
    # )
    
    tokenizer = utils.fn_load_tokenizer_chemberta(max_seq_length=config.MAX_SEQ_LENGTH) # Embed index error! XD but haven't added any tokens after calling this function. 
    # print(f"{tokenizer.pad_token_id}, {tokenizer.pad_token}")
    # and set vocab_size = len(tokenizer) Not tokenizer.vocab_size... But still error
    # tokenizer = utils.fn_load_tokenizer_chemllama(max_seq_length=config.MAX_SEQ_LENGTH)
    
    data_module = CustomDeepChemDataModule(
        df=df,
        tokenizer=tokenizer,
        max_seq_length=config.MAX_SEQ_LENGTH,
        train_size_ratio=config.TRAIN_SIZE_RATIO,
        batch_size_train=config.BATCH_SIZE_TRAIN,
        batch_size_valid=config.BATCH_SIZE_VALID,
        num_device=int(config.NUM_DEVICE) * int(config.NUM_WORKERS_MULTIPLIER),
        # num_device=int(config.NUM_DEVICE),
        # num_device=0
    )
    data_module.setup()

    # for warmup scheduler
    steps_per_epoch = len(data_module.train_dataloader()) ####

    model = ChemBerta(
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=config.PAD_TOKEN_ID,
        bos_token_id=config.BOS_TOKEN_ID,
        eos_token_id=config.EOS_TOKEN_ID,
        steps_per_epoch=steps_per_epoch, #
        warmup_epochs=config.MIN_EPOCHS, #
        max_epochs=config.MAX_EPOCHS, #
        hidden_size=config.HIDDEN_SIZE,
        intermediate_size=config.INTERMEDIATE_SIZE,
        num_labels=config.NUM_LABELS,
        attention_probs_dropout_prob=config.ATTENTION_PROBS_DROPOUT_PROB,
        hidden_dropout_prob=config.HIDDEN_DROPOUT_PROB,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        learning_rate=config.LEARNING_RATE
    )
    # print(model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath=config.DIR_CHECKPOINT, # specified on the Trainer side
        filename=config.NAME_MODEL + '_vloss_{val_loss:.3f}_ep_{epoch:02d}',
        every_n_epochs=1,
        save_top_k=-1,
    )

    # lr_monitor = LearningRateMonitor(logging_interval='step') ####

    # https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    trainer = L.Trainer(
        default_root_dir=config.DIR_CHECKPOINT,
        strategy="ddp",
        profiler=profiler,
        # profiler='simple',
        logger=csv_logger, ###
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES, # or 4
        # devices=1,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION, # precition float32
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, lr_monitor],
        # fast_dev_run=True, # sample run to check the code validity
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)