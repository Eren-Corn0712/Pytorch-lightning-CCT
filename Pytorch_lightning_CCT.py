import src
import wandb
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelSummary
import torchmetrics


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs[:self.num_samples],
                                               preds[:self.num_samples],
                                               val_labels[:self.num_samples])]
        })


class LogPredictionsCallback(Callback):

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 8
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)
            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)


def create_model(num_classes):
    return src.cct_7_3x2_32(pretrained=False, num_classes=num_classes)


class LitModel(LightningModule):
    def __init__(self, num_classes,
                 lr=5e-4,
                 weight_decay=5e-4):
        super().__init__()
        # Model setting
        self.model = create_model(num_classes)
        # loss
        self.loss = nn.CrossEntropyLoss()
        # optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        # metrics
        self.accuracy = torchmetrics.Accuracy()
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        '''defines model optimizer'''
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = self.accuracy(preds, y)
        return preds, loss, acc


if __name__ == '__main__':
    seed_everything(seed=2021)

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomChoice([
                torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
                torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            ]),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    cifar10_dm = CIFAR10DataModule(
        data_dir=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    cifar10_dm.prepare_data()
    cifar10_dm.setup()

    # Init our model
    model = LitModel(num_classes=cifar10_dm.num_classes, lr=5e-4, weight_decay=5e-4)

    # Initialize wandb logger
    wandb.login()
    wandb_logger = WandbLogger(name='Cifar10_CCT_model_20211228',
                               project='wandb-lightning',
                               job_type='train')

    wandb_logger.watch(model, log="gradients", log_graph=False)

    # Initialize a trainer
    trainer = Trainer(max_epochs=5000,
                      gpus=1,
                      logger=wandb_logger,
                      callbacks=[
                          LogPredictionsCallback(),
                          LearningRateMonitor(logging_interval="step"),
                          ModelSummary(),
                      ],
                      enable_checkpointing=True,
                      enable_progress_bar=True,
                      enable_model_summary=True)

    # training
    trainer.fit(model, cifar10_dm)
    # test
    trainer.test(model, datamodule=cifar10_dm)
    # Close wandb run
    wandb.finish()
