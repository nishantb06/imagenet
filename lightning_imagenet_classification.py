#!/usr/bin/env python
# coding: utf-8

### config

total_epochs = 100
batch_size = 256
num_processes = 2
image_size = 224
drop_path = 0.05
## Loss Function - CE (but try BCE)
# Always choose "SGD" for CNNs and AdamW for ViTs - SGD is Difficult to Converge || We should use LAMB with Cosine LR
## Multi-label --> Mixup and CutMix
LR = 5e-3
weight_decay = 0.05
warmup_epoch = 5
dropout = 0
drop_path = 0.05


# In[5]:


import wandb

wandb_token = "e653df8526c77d083379de033d13342620583fdf"

wandb.login(key=wandb_token)


# In[7]:


import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd


import albumentations

train_aug = albumentations.Compose(
    [
        albumentations.Resize(image_size, image_size, p=1),
        albumentations.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
        ),
        albumentations.OneOf(
            [
                albumentations.RandomGamma(gamma_limit=(90, 110)),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1
                ),
            ],
            p=0.5,
        ),
        albumentations.HorizontalFlip(),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.Resize(image_size, image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, augmentations=None, train=True):
        self.image_path = image_path
        self.augmentations = augmentations
        self.df = pd.read_csv(
            "/home/ubuntu/training/training/imagenet_class_labels.csv"
        )
        self.valid_df = pd.read_csv(
            "/home/ubuntu/training/training/validation_classes.csv"
        )
        self.train = train

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = np.asarray(image)

        ## center crop 95% area
        H, W, C = image.shape
        image = image[int(0.04 * H) : int(0.96 * H), int(0.04 * W) : int(0.96 * W), :]

        if self.train:
            class_id = str(self.image_path[item].split("/")[-2])
            targets = self.df[self.df["Index"] == class_id]["ID"].values[0] - 1
        else:
            class_id = str(self.image_path[item].split("/")[-1][:-5])
            targets = (
                self.valid_df[self.valid_df["ImageId"] == class_id]["LabelId"].values[0]
                - 1
            )

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


from timm.data.mixup import Mixup

mixup_args = {
    "mixup_alpha": 0.1,
    "cutmix_alpha": 1.0,
    "cutmix_minmax": None,
    "prob": 0.7,
    "switch_prob": 0,
    "mode": "batch",
    "label_smoothing": 0.1,
    "num_classes": 1000,
}
mixup_fn = Mixup(**mixup_args)


import glob
import random

train_paths = glob.glob(
    "/home/ubuntu/training/Imagenet/ILSVRC/Data/ImageNet/train/*/*.JPEG"
)
valid_paths = glob.glob(
    "/home/ubuntu/training/Imagenet/ILSVRC/Data/ImageNet/val/*.JPEG"
)


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


import torch
from timm import create_model
from torchvision import transforms, datasets
import pytorch_lightning as L

# from timm.scheduler.cosine_lr import CosineLRScheduler


class LitClassification(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model(
            "resnet50", pretrained=False, drop_path_rate=drop_path
        )
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, targets = batch["image"], batch["targets"]
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        acc1, acc5 = self.__accuracy(outputs, targets, topk=(1, 5))
        self.log("train_loss", loss)
        self.log(
            "train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True
        )
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch):
        images, targets = batch["image"], batch["targets"]
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)

        acc1, acc5 = self.__accuracy(outputs, targets, topk=(1, 5))
        self.log("valid_loss", loss)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=LR, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=warmup_epoch,
            steps_per_epoch=None,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
            last_epoch=-1,
            verbose="deprecated",
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ImageNetDataset(train_paths, train_aug, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_processes,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ImageNetDataset(valid_paths, valid_aug, train=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return valid_loader


L.seed_everything(879246)


wandb_logger = WandbLogger(log_model="all", project="ImageNet_Lightning")


# Initialize a trainer
best_checkpoint_callback = L.callbacks.ModelCheckpoint(
    filename="bestmodel-{epoch}-monitor-{val_acc1}", mode="max"
)
every_epoch_checkpoint_callback = L.callbacks.ModelCheckpoint(
    filename="{epoch}_{val_acc1}", every_n_epochs=10
)

trainer = L.Trainer(
    max_epochs=total_epochs,
    devices=torch.cuda.device_count(),
    accelerator="gpu",
    logger=wandb_logger,
    # callbacks=[early_stop_callback],
    precision=16,
    callbacks=[best_checkpoint_callback, every_epoch_checkpoint_callback],
)

model = LitClassification()

trainer.fit(
    model,
    ckpt_path="/home/ubuntu/training/training/ImageNet_Lightning/h94dnl2b/checkpoints/bestmodel-epoch=32-monitor-val_acc1=62.54399871826172.ckpt",
)
