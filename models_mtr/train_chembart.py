import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor ###
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler
import os

from dataset_mtr.utils import class_read_csv #1
from . import utils #2

from .chembart_mtr.chembart_mtr import ChemBart
from .datamodule_mtr import CustomDeepChemDataModule

# from .chembart_mtr import config_chembart_mtr_64b_med30m as config # g01
# from .chembart_mtr import config_chembart_mtr_64b_med20m as config # g04
# from .chembart_mtr import config_chembart_mtr_64b_med10m as config # g05
# from .chembart_mtr import config_chembart_mtr_64b_sm10m as config # g06
# from .chembart_mtr import config_chembart_mtr_64b_sm20m as config # g01
from .chembart_mtr import config_chembart_mtr_64b_sm30m as config # g03

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# (chemllm) [ylee@g002 transformers_and_chemistry]$ python -m models_mtr.train_chemllama
# python -m models_mtr.train_chemllama

# speed-up option
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
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

    tokenizer = utils.fn_load_tokenizer_chembart(max_seq_length=config.MAX_SEQ_LENGTH)

    data_module = CustomDeepChemDataModule(
        df=df,
        tokenizer=tokenizer,
        max_seq_length=config.MAX_SEQ_LENGTH,
        train_size_ratio=config.TRAIN_SIZE_RATIO,
        batch_size_train=config.BATCH_SIZE_TRAIN,
        batch_size_valid=config.BATCH_SIZE_VALID,
        num_device=int(config.NUM_DEVICE) * config.NUM_WORKERS_MULTIPLIER, # num_workers
        # num_device=0
    )
    data_module.setup()
    
    # for warmup scheduler
    steps_per_epoch = len(data_module.train_dataloader()) ####
    
    model = ChemBart(
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=config.PAD_TOKEN_ID,
        bos_token_id=config.BOS_TOKEN_ID,
        eos_token_id=config.EOS_TOKEN_ID,
        steps_per_epoch=steps_per_epoch, #
        warmup_epochs=config.MIN_EPOCHS, #
        max_epochs=config.MAX_EPOCHS, #
        d_model=config.D_MODEL,
        encoder_ffn_dim=config.ENCODER_FFN_DIM,
        decoder_ffn_dim=config.DECODER_FFN_DIM,
        encoder_layers=config.ENCODER_LAYERS,
        decoder_layers=config.DECODER_LAYERS,
        num_labels=config.NUM_LABELS,
        dropout=config.DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT,
        learning_rate=config.LEARNING_RATE,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath=config.DIR_CHECKPOINT, # specified on the Trainer side
        filename=config.NAME_MODEL + '_vloss_{val_loss:.3f}_ep_{epoch:02d}',
        every_n_epochs=1,
        save_top_k=-1,
    )
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    # https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    trainer = L.Trainer(
        default_root_dir=config.DIR_CHECKPOINT,
        strategy="ddp",
        profiler=profiler,
        logger=csv_logger, ###
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES, # or 4
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION, # precition float32
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, lr_monitor],
        # fast_dev_run=True, # sample run to check the code validity
        # enable_checkpointing=False,
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)