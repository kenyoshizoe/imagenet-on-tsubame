from lightning.pytorch import LightningDataModule
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import webdataset as wds


class RepeatedWebDataset(wds.WebDataset):
    def __init__(self, *args, repetitions=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.repetitions = repetitions

    def __iter__(self):
        for _ in range(self.repetitions):
            for sample in super().__iter__():
                yield sample


class ImageNet1KWDSDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self._train_filenames = [f"imagenet1k-train-{sid:04d}.tar" for sid in range(1024)]
        # self._val_filenames = [f"imagenet1k-validation-{sid:02d}.tar" for sid in range(64)]
        self._train_filenames = [f"imagenet1k-train-{sid:04d}.tar" for sid in range(2)]
        self._val_filenames = [f"imagenet1k-validation-{sid:02d}.tar" for sid in range(2)]

    def setup(self, stage=None):
        train_shards = []
        for f in self._train_filenames:
            train_shards.append(f"file:{self.cfg.path}/{f}")
        self.train_dataset = (
            RepeatedWebDataset(train_shards, shardshuffle=True, nodesplitter=wds.split_by_node, empty_check=False)
            .shuffle(1000)
            .decode("pil")
            .map_dict(jpg=instantiate(self.cfg.train.transform))
            .to_tuple("jpg", "json")
        )

        valid_shards = []
        for f in self._val_filenames:
            valid_shards.append(f"file:{self.cfg.path}/{f}")
        self.valid_dataset = (
            RepeatedWebDataset(valid_shards, shardshuffle=True, nodesplitter=wds.split_by_node, empty_check=False)
            .decode("pil")
            .map_dict(jpg=instantiate(self.cfg.valid.transform))
            .to_tuple("jpg", "json")
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            drop_last=True
        )
