import lightning as L
import torch
import torchmetrics

from torch import nn
from transformers import LlamaModel, LlamaConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class ChemLlama(L.LightningModule):
    def __init__(
        self,
        max_position_embeddings,
        vocab_size,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        steps_per_epoch=None, #
        warmup_epochs=None, #
        max_epochs=None, #
        hidden_size=384,
        intermediate_size=464,
        num_labels=105,
        attention_dropout=0.144,
        num_hidden_layers=3,
        num_attention_heads=12,
        learning_rate=0.0001,
    ):
        super(ChemLlama, self).__init__()
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.steps_per_epoch = steps_per_epoch #
        self.warmup_epochs = warmup_epochs #
        self.max_epochs = max_epochs #
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.learning_rate = learning_rate

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

        self.config_llama = LlamaConfig(
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        self.loss_fn = nn.L1Loss()

        self.llama = LlamaModel(self.config_llama)
        self.gelu = nn.GELU()
        self.score = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):

        transformer_outputs = self.llama(
            input_ids=input_ids, attention_mask=attention_mask
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.gelu(hidden_states)
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config_llama.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config_llama.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config_llama.pad_token_id).int().argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        # raise ValueError(len(sequence_lengths), sequence_lengths)

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        return pooled_logits

    def training_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)

        # mae = self.mae(logits, labels)
        # mse = self.mse(logits, labels)
        self.log_dict(
            {
                "train_loss": loss, 
                # "train_mae": mae, 
                # "train_mse": mse
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            # logger=True,
        )
        # on_stecp = True will use lots of computational resources

        # return loss
        return {"loss": loss, "logits": logits, "labels": labels}

    def train_epoch_end(self, outputs):
        # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.log_dict(
            {
                "train_mae": self.mae(scores, labels),
                "train_mse": self.mse(scores, labels)
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)
        # self.log("val_loss", loss)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):

        loss, logits, labels = self._common_step(batch=batch, batch_idx=batch_idx)
        # self.log("val_loss", loss)
        self.log("test_loss", loss, sync_dist=True,)
        return loss

    def _common_step(self, batch, batch_idx):

        logits = self.forward(
            input_ids=batch["input_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        labels = batch["labels"].squeeze()
        loss = self.loss_fn(logits, labels)

        # print(f"logits : {logits.shape}")
        # print(f"labels : {labels.shape}")

        return loss, logits, labels

    # def configure_optimizers(self):  # Schedular here too!
    #     # since confiture_optimizers and the model are included in the same class.. self.parameters()
    #     return torch.optim.AdamW(
    #         params=self.parameters(),
    #         lr=self.learning_rate,
    #         betas=(0.9, 0.999),
    #         weight_decay=0.01,
    #     )

    # # The below is for warm-up scheduler
    # https://lightning.ai/forums/t/how-to-use-warmup-lr-cosineannealinglr-in-lightning/1980
    # https://github.com/Lightning-AI/pytorch-lightning/issues/328
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
            warmup_epochs=self.warmup_epochs*self.steps_per_epoch // torch.cuda.device_count(), # // num_device in this case
            max_epochs=self.max_epochs*self.steps_per_epoch // torch.cuda.device_count(),
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