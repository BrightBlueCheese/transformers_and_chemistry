import lightning as L
import torch
import torchmetrics

from torch import nn
from transformers import BartModel, BartConfig

# from torch.optim.lr_scheduler import CosineAnnealingLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from . import config_chembart_mtr_64

class ChemBart(L.LightningModule):
    def __init__(
        self,
        max_position_embeddings,
        vocab_size,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        steps_per_epoch, #
        warmup_epochs, #
        max_epochs, #
        encoder_ffn_dim=384,
        decoder_ffn_dim=384,
        d_model=464,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_layers=3,
        decoder_layers=3,
        num_labels=105, # not for the default BartConfig bur for the custom one
        dropout=0.144, # default was 0 so a s Llama but not RoBerta
        attention_dropout=0.144,
        learning_rate=0.0001,
    ):
        super(ChemBart, self).__init__()
        self.save_hyperparameters()

        # not necessarily come with self, but I wanted it to
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.steps_per_epoch = steps_per_epoch #
        self.warmup_epochs = warmup_epochs #
        self.max_epochs = max_epochs #
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.d_model = d_model
        self.encoder_attention_heads=encoder_attention_heads
        self.decoder_attention_heads=decoder_attention_heads
        self.encoder_layers=encoder_layers
        self.decoder_layers=decoder_layers
        self.num_labels = num_labels
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.learning_rate = learning_rate
    
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

        self.config_bart = BartConfig(
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.eos_token_id,
            forced_eos_token_id=self.eos_token_id,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            d_model=self.d_model,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )

        self.loss_fn = nn.L1Loss()

        self.bart = BartModel(self.config_bart)
        self.gelu = nn.GELU()
        self.score = nn.Linear(self.decoder_ffn_dim, self.num_labels)

    # https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/bart/modeling_bart.py#L1916
    def forward(self, input_ids, attention_mask, labels=None):

        transformer_outputs = self.bart(
            input_ids=input_ids, attention_mask=attention_mask
        )

        hidden_states = transformer_outputs[0]
        eos_mask = input_ids.eq(self.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        
        sentence_representation = self.gelu(sentence_representation)
        logits = self.score(sentence_representation)

        return logits

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
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.num_steps)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            # warmup_epochs=self.warmup_epochs*self.steps_per_epoch // 4, # // num_device in this case
            # max_epochs=self.max_epochs*self.steps_per_epoch // 4,
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