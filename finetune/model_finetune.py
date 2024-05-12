import torch
from torch import nn
import lightning as L
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import numpy as np

class CustomFinetuneModel(L.LightningModule):
    def __init__(
        self,
        model_mtr,
        steps_per_epoch, #
        warmup_epochs, #
        max_epochs, #
        learning_rate,
        dataset_dict:dict,
        linear_param:int=64,
        use_freeze:bool=True,
        *args, **kwargs
    ):
        super(CustomFinetuneModel, self).__init__()
        # self.save_hyperparameters()

        self.model_mtr = model_mtr
        if use_freeze:
            self.model_mtr.freeze()
            # for name, param in model_mtr.named_parameters():
            #     param.requires_grad = False
            #     print(name, param.requires_grad)

        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        self.list_val_loss = list()
        
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(self.model_mtr.num_labels, linear_param)
        self.linear2 = nn.Linear(linear_param, linear_param)
        self.regression = nn.Linear(linear_param, 1)

        if dataset_dict['dataset_type'] == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif dataset_dict['dataset_type'] == 'regression':
            self.loss_fn = nn.L1Loss()
        else :
            return ValueError(f"Check the dataset type. {dataset_dict['dataset_name']} : We got {dataset_dict['dataset_type']}.")

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.model_mtr(input_ids=input_ids, attention_mask=attention_mask)
        x = self.gelu(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.regression(x)
        
        return x

    def training_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)

        self.log_dict(
            {
                "train_loss": loss, 
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True,
        )

        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)
        # self.log("val_loss", loss)
        self.log("val_loss", loss, sync_dist=True)
        
        return loss

    def valid_epoch_end(self, outputs):
        # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.list_val_loss.append(self.loss_fn(scores, labels))
        self.log_dict(
            {
                "list_val_loss": self.list_val_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True,
        )
    
    # def get_val_loss_history(self):
    #     return np.array(self.list_val_loss).squeeze()
        
    def test_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)
        # self.log("val_loss", loss)
        self.log("test_loss", loss, sync_dist=True,)
        
        return loss

    def _common_step(self, batch, batch_idx):

        logits = self.forward(
            input_ids=batch["input_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        ).squeeze()

        # labels = batch["labels"].squeeze()
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        
        # print(f"logits : {logits.shape} | labels : {labels.shape}")
        # print(f"labels : {labels.shape}")

        return loss, logits, labels

    def predict_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)
        
        return logits, labels

    def configure_optimizers(self):  # Schedular here too!
        # since confiture_optimizers and the model are included in the same class.. self.parameters()
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # "warmup_epochs //4 only not max_epochs" will work
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            # warmup_epochs=self.warmup_epochs*self.steps_per_epoch // 4, # // num_device in this case
            # max_epochs=self.max_epochs*self.steps_per_epoch // 4,
            # Better not to use Multiple GPUs due to the smaller dataset size.
            warmup_epochs=self.warmup_epochs*self.steps_per_epoch, # // num_device in this case
            max_epochs=self.max_epochs*self.steps_per_epoch,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "val_loss",
            }
        }

    
        