import lightning as L
import torch
from dataset_mtr import utils

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class CustomDeepChemDatasetLazy(Dataset):
    def __init__(self, df, tokenizer, max_seq_length):
        # self.target_df = target_df
        self.keys = df.iloc[:, 0]
        self.labels = df.iloc[:, 1:]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.keys.shape[0]

    def fn_token_encode(self, smiles):
        return self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
        )

    def __getitem__(self, idx):
        local_encoded = self.fn_token_encode(self.keys.iloc[idx])

        return {
            "input_ids": torch.tensor(local_encoded["input_ids"]),
            "attention_mask": torch.tensor(local_encoded["attention_mask"]),
            "labels": torch.tensor(self.labels.iloc[idx]),
        }


# https://www.youtube.com/watch?v=e47f__x7KSE
class CustomDeepChemDataModule(L.LightningDataModule):
    def __init__(
        self,
        df,
        tokenizer,
        max_seq_length,
        train_size_ratio,
        batch_size_train,
        batch_size_valid,
        num_device,
    ):
        super().__init__()

        # self.target_keys = target_df.iloc[:, 0]
        # self.target_labels = target_df.iloc[:, 1:]
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.index_to_split = int(self.df.shape[0] * train_size_ratio)
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.num_device = num_device
        # self.pin_memory = True if self.num_device > 0 else False
    
        # self.df_train = self.df.iloc[: self.index_to_split, :]
        # self.df_valid = self.df.iloc[self.index_to_split :, :]

    def prepare_data(self):
        # single gpu
        pass

    def setup(self, stage=None):
        # multiple gpu
        self.df_train = self.df.iloc[: self.index_to_split, :]
        self.df_valid = self.df.iloc[self.index_to_split :, :]

    def train_dataloader(self):
        return DataLoader(
            dataset=CustomDeepChemDatasetLazy(
                self.df_train, self.tokenizer, self.max_seq_length
            ),
            batch_size=self.batch_size_train,
            num_workers=self.num_device,
            # pin_memory=self.pin_memory,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=CustomDeepChemDatasetLazy(
                self.df_valid, self.tokenizer, self.max_seq_length
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            # pin_memory=self.pin_memory,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        pass
