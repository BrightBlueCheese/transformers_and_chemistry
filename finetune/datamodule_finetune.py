import lightning as L
import torch
import deepchem as dc
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding


# class CustomLlamaDatasetMolnet(Dataset):
#     def __init__(self, target_encodings, target_labels):
#         self.encodings = target_encodings
#         self.labels = target_labels

#     def __len__(self):
#         return len(self.encodings['input_ids'])

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

class CustomLlamaDatasetMolnet(Dataset):
    def __init__(self, df, tokenizer, max_seq_length):
        self.keys = df.ids # 1D array
        self.labels = df.y[: ,-1] # 2D array
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
        local_encoded = self.fn_token_encode(self.keys[idx])
        
        return {
            "input_ids": torch.tensor(local_encoded["input_ids"]),
            "attention_mask": torch.tensor(local_encoded["attention_mask"]),
            "labels": torch.tensor(self.labels[idx]),
        }

def load_finetune_dataset(dataset_dict):
    try:
        specific_task = dataset_dict['specific_task']
    except:
        specific_task = None

    tasks, df, _ = dataset_dict['load_fn'](featurizer="Raw", 
                                           splitter=dataset_dict['split'])
    
    if specific_task == None or (specific_task != None and len(tasks) == 1):
        return tasks, df
        
    elif specific_task != None and len(tasks) != 1:
        tasks_label_index = tasks.index(specific_task[0])
        
        list_df = list()
        for local_df in df:
            local_df = dc.data.DiskDataset.from_numpy(local_df.X,
                                                      np.expand_dims(local_df.y[:, tasks_label_index], axis=1),
                                                      np.expand_dims(local_df.w[:, tasks_label_index], axis=1),
                                                      local_df.ids,
                                                      specific_task)
            list_df.append(local_df)
        return specific_task, list_df
            
    else:
        raise ValueError("Please check the molnet dictionary. Something is wrong.")




# def load_finetune_dataloader(dataset_dict:dict, 
#                              tokenizer, 
#                              truncation:bool=True, 
#                              padding='max_length', 
#                              max_length:int=512, 
#                              batch_size_train:int=32, 
#                              batch_size_valid:int=int(32*1.5)):

#     tasks, df = load_finetune_dataset(dataset_dict)
    
#     print(f"Loading dataloaders for '{dataset_dict['dataset_name']}'")
#     print("=======================================")
#     print(f"The shape of the train, valid, test datasets : {df[0].y.shape}, {df[1].y.shape}, {df[1].y.shape}")
    
#     list_dataset = list()
#     for local_df in df:
#         local_encodings = tokenizer(local_df.ids.tolist(), truncation=truncation, padding=padding, max_length=max_length)
#         local_dataset = CustomLlamaDatasetMolnet(local_encodings, np.expand_dims(local_df.y[:, -1], axis=1))
#         list_dataset.append(local_dataset)

    

#     data_collator = DataCollatorWithPadding(tokenizer)

#     # train, valid, test
#     list_batch_size = [batch_size_train, batch_size_valid, batch_size_valid]
#     list_shuffle = [True, False, False]
    
#     list_dataloader = list()
#     for local_dataset, local_batch_size, local_shuffle in zip(list_dataset, list_batch_size, list_shuffle):
#         local_dataloader = DataLoader(local_dataset, batch_size=local_batch_size, shuffle=local_shuffle, collate_fn=data_collator)
#         list_dataloader.append(local_dataloader)

#     for batch in list_dataloader[0]:
#         break
#     temp_dict = {k: v.shape for k, v in batch.items()}
#     print(f"First train batch:\n{temp_dict}")

#     for batch in list_dataloader[1]:
#         break
#     temp_dict = {k: v.shape for k, v in batch.items()}
#     print(f"First valid & test batch:\n{temp_dict}")
#     print("=======================================")

#     return list_dataloader


class CustomFinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dict,
        tokenizer,
        max_seq_length,
        batch_size_train,
        batch_size_valid,
        num_device,
    ):
        super().__init__()

        self.dataset_dict = dataset_dict
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.num_device = num_device
    

    def prepare_data(self):
        _, self.list_df = load_finetune_dataset(self.dataset_dict)

    def setup(self, stage=None):
        self.train_df, self.valid_df, self.test_df = self.list_df

    def train_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetMolnet(
                self.train_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_train,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetMolnet(
                self.valid_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=CustomLlamaDatasetMolnet(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        
    # It uses test_df
    def predict_dataloader(self): 
        return DataLoader(
            dataset=CustomLlamaDatasetMolnet(
                self.test_df, self.tokenizer, self.max_seq_length,
            ),
            batch_size=self.batch_size_valid,
            num_workers=self.num_device,
            collate_fn=self.data_collator,
            shuffle=False,
        )








        