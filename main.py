import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig
import importlib
import os
import torch


@hydra.main(config_path="./config", config_name="imagenet_1k", version_base=None)
def main(config: DictConfig):
    if 'seed' in config:
        pl.seed_everything(config.seed, workers=True)  # random, numpy, torch
        # additional seed settings
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    loggers = []
    for logger in config.trainer.logger:
        if logger == 'wandb':
            loggers.append(pl.loggers.WandbLogger(
                save_dir='./logs',
                project=config.project_name,
                name=config.run_name
            ))
        elif logger == 'csv':
            loggers.append(pl.loggers.CSVLogger(
                save_dir='./logs',
                name=config.run_name
            ))
        else:
            raise ValueError('Invalid logger name')

    callbacks = []
    for callback in config.trainer.callbacks:
        callbacks.append(hydra.utils.instantiate(callback))

    trainer = pl.Trainer(logger=loggers,
                         max_epochs=config.trainer.max_epochs,
                         min_epochs=config.trainer.min_epochs,
                         max_steps=config.trainer.max_steps,
                         min_steps=config.trainer.min_steps,
                         num_sanity_val_steps=config.trainer.num_sanity_val_steps if 'num_sanity_val_steps' in config.trainer else 2,
                         callbacks=callbacks,
                         default_root_dir='./logs',
                         use_distributed_sampler=config.trainer.use_distributed_sampler if 'use_distributed_sampler' in config.trainer else True)

    if config.ckpt_path:
        module_path, class_name = config.model._target_.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass.load_from_checkpoint(config.ckpt_path)
    else:
        model = hydra.utils.instantiate(config.model, _recursive_=False)

    datamodule = None
    if 'datamodule' in config:
        datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # test best model
    if config.mode == 'train_test':
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule, ckpt_path='best', verbose=True)
    elif config.mode == 'train':
        trainer.fit(model, datamodule=datamodule)
    elif config.mode == 'test':
        trainer.test(model, datamodule=datamodule, verbose=True)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    main()
