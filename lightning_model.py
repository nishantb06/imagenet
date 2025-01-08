import torch
import pytorch_lightning as L
from timm import create_model

class LitClassification(L.LightningModule):
    def __init__(self, drop_path=0.05):
        super().__init__()
        self.model = create_model(
            "resnet50", pretrained=False, drop_path_rate=drop_path
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
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