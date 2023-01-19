from typing import List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from kornia.geometry import resize
from pytorch_toolbelt.losses import BinaryFocalLoss
from torchmetrics.classification.jaccard import JaccardIndex
from torchvision.transforms.functional import InterpolationMode


class ImageSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        segmentation_model: dict,
        intermediate_shape: List[int] = [256, 256],
        lr: float = 1.0e-4,
        betas: List[float] = [0.9, 0.999],
        milestones: List[int] = [0],
        gamma: float = 1.0,
    ):
        super().__init__()

        self._model = getattr(smp, segmentation_model["name"])(
            **segmentation_model["kwargs"]
        )

        self._intermediate_shape = intermediate_shape

        self.lr = lr  # https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html#using-lightning-s-built-in-lr-finder
        self._betas = betas

        self._milestones = milestones
        self._gamma = gamma

        self._criterion = BinaryFocalLoss()
        self._jaccard = JaccardIndex(
            task="multiclass", num_classes=segmentation_model["kwargs"]["classes"] + 1
        )

    def forward(self, img):
        shape = img.shape[-2:]
        if self.training:
            img = resize(img, self._intermediate_shape, align_corners=False)
            seg = self._model(img)
            seg = resize(seg, shape, align_corners=False)
        else:
            img = resize(img, self._intermediate_shape, align_corners=False)
            seg = torch.sigmoid(self._model(img))
            seg = resize(seg, shape, align_corners=False)
            seg = (seg > 0.5).float()
        return seg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, betas=self._betas
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self._milestones, gamma=self._gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, seg = batch

        # predict
        seg_pred = self(img)
        bfl = self._criterion(seg_pred, seg)

        self.log("train/binary_focal_loss", bfl)

        return bfl

    def validation_step(self, batch, batch_idx):
        img, seg = batch

        # predict
        seg_pred = self(img)
        bfl = self._criterion(seg_pred, seg)
        iou = self._jaccard(seg_pred, seg.int())

        self.log_dict({"val/binary_focal_loss": bfl, "val/iou": iou})

    def test_step(self, batch, batch_idx):
        img, seg = batch

        # predict
        seg_pred = self(img)
        bfl = self._criterion(seg_pred, seg)
        iou = self._jaccard(seg_pred, seg.int())

        # log images
        self.logger.experiment.add_images("test/seg", seg, 0)
        self.logger.experiment.add_images("test/img", img, 0)
        self.logger.experiment.add_images("test/seg_pred", seg_pred, 0)

        self.log_dict({"test/binary_focal_loss": bfl, "test/iou": iou})
