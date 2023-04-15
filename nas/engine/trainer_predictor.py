# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import random
from torch_geometric.data import Batch
from nas.utils.logger import MetricLogger
from nas.engine.build import TRAINER_REGISTRY
from nas.utils.solver import gen_batch_idx, CosineLR, make_agent_optimizer
from nas.predictors.build import build_predictor
from nas.utils.predictors import load_predictor_ccl
import logging


@TRAINER_REGISTRY.register()
class GINPredictorTrainer:
    def __init__(self, cfg, device, num_architectures):
        self.cfg = cfg
        self.logger = logging.getLogger(f"nas_gpu_{device}")

        self.criterion = self.__get_cirterion()
        self.device = device
        self.num_architectures = num_architectures
        self.predictor = build_predictor(cfg, device)
        load_pretrained = True if len(cfg.PREDICTOR.RESUME_DIR) > 0 else False
        self.optimizer = make_agent_optimizer(self.predictor, base_lr=cfg.SOLVER_NAS.LR,
                                              weight_deacy=cfg.SOLVER_NAS.WEIGHT_DEACY,
                                              bias_multiply=True)
        if load_pretrained:
            self.predictor = self.load_model()
            self.logger.info(f'load model from path {self.cfg.PREDICTOR.RESUME_DIR}')
            self.predictor.fc = torch.nn.Linear(cfg.PREDICTOR.DIM2, cfg.PREDICTOR.NUM_CLASSES, bias=True)
            torch.nn.init.kaiming_uniform_(self.predictor.fc.weight, a=1)
            self.predictor.fc.bias.data.zero_()

        self.predictor.to(self.device)
        self.scheduler = CosineLR(self.optimizer,
                                  epochs=cfg.SOLVER_NAS.EPOCHS,
                                  train_images=self.num_architectures,
                                  batch_size=cfg.SOLVER_NAS.BATCH_SIZE)
        self.batch_size = cfg.SOLVER_NAS.BATCH_SIZE
        self.epoch = cfg.SOLVER_NAS.EPOCHS
        self.rate = cfg.SOLVER_NAS.RATE

    def fit(self, all_g_data, val_accuracy, logger=None):
        meters = MetricLogger(delimiter=" ")
        self.predictor.train()
        for epoch in range(self.epoch):
            idx_list = list(range(len(all_g_data)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, self.batch_size)
            counter = 0
            for i, batch_idx in enumerate(batch_idx_list):
                counter += len(batch_idx)
                data_list = []
                target_list = []
                for idx in batch_idx:
                    data_list.append(all_g_data[idx])
                    target_list.append(val_accuracy[idx])
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch

                if self.cfg.PREDICTOR.TYPE == 'GINPredictor':
                    pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx)*self.rate
                    pred = pred.squeeze()
                    loss = self.criterion(pred, val_tensor)
                else:
                    raise NotImplementedError()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                meters.update(loss=loss.item())
        if logger:
            logger.info(f'Neural Predictor Avg Loss: {meters.meters["loss"].avg}, '
                        # f'Global Avg Loss: {meters.meters["loss"].global_avg}, '
                        f'Median Value: {meters.meters["loss"].median}')
        return meters.meters['loss'].avg

    def pred(self, all_g_data):
        if self.cfg.PREDICTOR.TYPE == 'GINPredictor':
            return self.pred_predictor(all_g_data)
        else:
            raise NotImplementedError(f"Predictor {self.cfg.PREDICTOR.TYPE} does not support at present!")

    def pred_predictor(self, all_g_data):
        pred_list = []
        idx_list = list(range(len(all_g_data)))
        self.predictor.eval()
        batch_idx_list = gen_batch_idx(idx_list, 64)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = []
                for idx in batch_idx:
                    data_list.append(all_g_data[idx])
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx).squeeze() * self.rate

                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def __get_cirterion(self):
        if self.cfg.PREDICTOR.TYPE == 'GINPredictor':
            return torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'Predictor {self.cfg.PREDICTOR.TYPE} has not implement!')

    def update_epoch(self, epoch):
        self.epoch = epoch

    def load_model(self):
        model_path = self.cfg.PREDICTOR.RESUME_DIR
        model_type = self.cfg.PREDICTOR.PRE_TRAIN_METHOD

        if model_type == 'SS_CCL':
            load_predictor_ccl(self.predictor, model_path)
        else:
            raise NotImplementedError(f"Predictor pre_train method {model_type} does not support at present!")
