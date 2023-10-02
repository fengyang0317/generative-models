from typing import Optional

import datasets
import torchdata.datapipes.iter
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
import torch
import torchvision.transforms as transforms
import functools
from torchvision.transforms.functional import InterpolationMode


def transform_fn(examples, transform):
  examples['image'] = [transform(i.convert('RGB')) for i in examples['image']]
  del examples['filepath']
  return examples


class StableDataModuleFromConfig(LightningDataModule):
  def __init__(
      self,
      train: DictConfig,
      validation: Optional[DictConfig] = None,
      test: Optional[DictConfig] = None,
      skip_val_loader: bool = False,
  ):
    super().__init__()
    self.train_config = train

    self.val_config = validation
    if not skip_val_loader:
      if self.val_config is None:
        print(
          "Warning: No validation dataset defined, using that one from training"
        )
        self.val_config = train
    self.test_config = test

  def setup(self, stage: str) -> None:
    print("Preparing datasets")
    transform = transforms.Compose(
      [
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
          mean=0.5, std=0.5
        ),
      ]
    )

    self.train_ds = datasets.load_from_disk(**self.train_config.dataset)
    self.train_ds.set_transform(
      functools.partial(transform_fn, transform=transform))
    if self.val_config:
      self.val_ds = datasets.load_from_disk(**self.val_config.dataset)
      self.val_ds.set_transform(
        functools.partial(transform_fn, transform=transform))
    if self.test_config:
      self.test_ds = datasets.load_from_disk(**self.test_config.dataset)
      self.test_ds.set_transform(
        functools.partial(transform_fn, transform=transform))

  def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
    loader = torch.utils.data.DataLoader(self.train_ds,
                                         **self.train_config.loader)
    return loader

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_ds, **self.val_config.loader)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_ds, **self.test_config.loader)
