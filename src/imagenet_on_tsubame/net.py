import torch
from lightning.pytorch import LightningModule
import hydra


class Net(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # backbone
        self.backbone = hydra.utils.instantiate(config.backbone.model)
        # head
        for head in config.heads:
            # Layer
            setattr(self, f'{head.name}_layer', hydra.utils.instantiate(head.layer))
            # Loss
            setattr(self, f'{head.name}_loss', hydra.utils.instantiate(head.loss))
            # Metric
            for metric in head.metrics:
                setattr(self, f'{head.name}_{metric.name}_train', hydra.utils.instantiate(metric.metric))
                setattr(self, f'{head.name}_{metric.name}_valid', hydra.utils.instantiate(metric.metric))
                setattr(self, f'{head.name}_{metric.name}_test', hydra.utils.instantiate(metric.metric))
        self.save_hyperparameters(config)

    def forward(self, x):
        x = self.backbone(x)

        preds = {}
        for head in self.config.heads:
            fc = getattr(self, f'{head.name}_layer')
            pred = fc(x)
            preds[head.name] = pred
        return preds

    def training_step(self, batch, batch_idx):
        total_loss, losses, metrics = self._get_preds_loss_acc(batch, mode='train')
        for key, value in losses.items():
            self.log(f'train/{key}', value, batch_size=batch[0].size(0))
        self.log('train/loss', total_loss, batch_size=batch[0].size(0))
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', lr, batch_size=batch[0].size(0))
        return total_loss

    def on_train_epoch_end(self):
        for head in self.config.heads:
            for metric in head.metrics:
                m = getattr(self, f'{head.name}_{metric.name}_train')
                v = m.compute()
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        self.log(f'train/{head.name}_{metric.name}', v)
                    elif v.dim() == 1:
                        for i, vv in enumerate(v):
                            self.log(f'train/{head.name}_{metric.name}_{i}', vv)
                    else:
                        raise ValueError('Logging metric with dim > 1 is not supported')
                else:
                    self.log(f'train/{head.name}_{metric.name}', v)
                m.reset()

    def validation_step(self, batch, batch_idx):
        total_loss, losses, metrics = self._get_preds_loss_acc(batch, mode='valid')
        for key, value in losses.items():
            self.log(f'valid/{key}', value, batch_size=batch[0].size(0))
        self.log('valid/loss', total_loss, batch_size=batch[0].size(0))

    def on_validation_epoch_end(self):
        for head in self.config.heads:
            for metric in head.metrics:
                m = getattr(self, f'{head.name}_{metric.name}_valid')
                v = m.compute()
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        self.log(f'valid/{head.name}_{metric.name}', v)
                    elif v.dim() == 1:
                        for i, vv in enumerate(v):
                            self.log(f'valid/{head.name}_{metric.name}_{i}', vv)
                    else:
                        raise ValueError('Logging metric with dim > 1 is not supported')
                else:
                    self.log(f'valid/{head.name}_{metric.name}', v)
                m.reset()

    def test_step(self, batch, batch_idx):
        total_loss, losses, metrics = self._get_preds_loss_acc(batch, mode='test')
        for key, value in losses.items():
            self.log(f'test/{key}', value, batch_size=batch[0].size(0))
        self.log('test/loss', total_loss, batch_size=batch[0].size(0))

    def on_test_epoch_end(self):
        for head in self.config.heads:
            for metric in head.metrics:
                m = getattr(self, f'{head.name}_{metric.name}_test')
                v = m.compute()
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        self.log(f'test/{head.name}_{metric.name}', v)
                    elif v.dim() == 1:
                        for i, vv in enumerate(v):
                            self.log(f'test/{head.name}_{metric.name}_{i}', vv)
                    else:
                        raise ValueError('Logging metric with dim > 1 is not supported')
                else:
                    self.log(f'test/{head.name}_{metric.name}', v)
                m.reset()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
        scheduler = None
        if 'lr_scheduler' in self.config:
            scheduler = {
                'scheduler': hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer),
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        return optimizer

    def _get_preds_loss_acc(self, batch, mode='train'):
        if mode not in ['train', 'valid', 'test']:
            raise ValueError('Invalid mode')

        data, label = batch
        preds = self(data)

        total_loss = 0
        losses = {}
        metrics = {}

        for head in self.config.heads:
            pred = preds[head.name].squeeze()
            # calc loss
            task_label = label[head.label_key]
            if 'label_type' in head:
                if head.label_type == 'float':
                    task_label = task_label.float()
                elif head.label_type == 'int':
                    task_label = task_label.long()
                else:
                    raise ValueError('Invalid label_type')

            loss_func = getattr(self, f'{head.name}_loss')
            task_loss = loss_func(pred, task_label)
            if "weight" in head:
                task_loss *= head.weight
            total_loss += task_loss
            losses[f'{head.name}_loss'] = task_loss
            # calc metric
            for metric in head.metrics:
                if 'target_key' not in metric:
                    target_key = head.label_key
                else:
                    target_key = metric.target_key

                m = getattr(self, f'{head.name}_{metric.name}_{mode}')
                target = label[target_key]
                metrics[f'{head.name}_{metric.name}_{mode}'] = m(pred, target)
        return total_loss, losses, metrics
