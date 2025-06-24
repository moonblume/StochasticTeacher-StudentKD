import numpy as np
import torch
from numpy import inf
from trainers.metrics_manager import MetricTracker, accuracy, f1, _calc_metrics
from trainers.checkpoint_handler import _save_checkpoint
from trainers.device_prep import _prepare_device

selected_d = {"outs": [], "trg": []}

def at_loss(student_feature, teacher_feature):
    return torch.nn.functional.mse_loss(student_feature, teacher_feature)

class Trainer:
    def __init__(self, model, loss, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, teacher_model=None, at_loss_weight=0.2):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.fold_id = fold_id
        self.device, device_ids = _prepare_device(self.logger, config['n_gpu'])

        self.checkpoint_dir = config.save_dir

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.loss = loss
        self.metric_ftns = [eval(metric) for metric in config['metrics']]
        self.optimizer = optimizer

        config_trainer = config['trainer']
        self.epochs = config_trainer['epochs']
        self.save_period = config_trainer['save_period']
        self.monitor = config_trainer.get('monitor', 'off')
        self.start_epoch = 1

        self._setup_monitoring()

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(data_loader.batch_size) * 1

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.curr_best = 0

        self.teacher_model = teacher_model
        self.at_loss_weight = at_loss_weight

    def _setup_monitoring(self):
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.config['trainer'].get('early_stop', inf)

    def train(self):
        not_improved_count = 0
        all_outs = []
        all_trgs = []

        for epoch in range(self.start_epoch, self.epochs + 1):
            result, epoch_outs, epoch_trgs = self._train_epoch(epoch, self.epochs)

            log = {'epoch': epoch}
            log.update(result)
            all_outs.extend(epoch_outs)
            all_trgs.extend(epoch_trgs)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                _save_checkpoint(self.model, self.optimizer, epoch, self.mnt_best, self.config,
                                 self.checkpoint_dir, self.logger, save_best=True)

        outs_name = "outs_" + str(self.fold_id)
        trgs_name = "trgs_" + str(self.fold_id)
        np.save(self.config._save_dir / outs_name, all_outs)
        np.save(self.config._save_dir / trgs_name, all_trgs)

        _calc_metrics(self.config, self.config._save_dir, fold_id=self.fold_id)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            _calc_metrics(self.config, self.checkpoint_dir)

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # self.optimizer.zero_grad()
            # output, student_feature = self.model(data)
            # student_feature, output = self.model.extract_features(data)
            


            # loss = self.loss(output, target)

            # if self.teacher_model is not None:
            #     with torch.no_grad():
            #         _, teacher_feature = self.teacher_model(data)
            #     att_transfer_loss = at_loss(student_feature, teacher_feature)
            #     loss += self.at_loss_weight * att_transfer_loss

            # loss = ((1-self.at_loss_weight)*gt_loss) + (self.at_loss_weight * att_transfer_loss) 

            # loss.backward()
            # self.optimizer.step()

        #     self.train_metrics.update('loss', loss.item())
        #     for met in self.metric_ftns:
        #         self.train_metrics.update(met.__name__, met(output, target))

        #     if batch_idx % self.log_step == 0:
        #         self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
        #             epoch,
        #             self._progress(batch_idx),
        #             loss.item(),
        #         ))

        #     if batch_idx == self.len_epoch:
        #         break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.curr_best:
                self.curr_best = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            if epoch == 10:
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-4

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
