#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


def generate(csv_path):
    hdfs = {}
    length = 0
    f = pd.read_csv(csv_path)
    log_key_seq_str = " ".join([str(EventId) for EventId in f["EventId"]])
    line = tuple(map(lambda n: n - 1, map(int, log_key_seq_str.strip().split())))
    hdfs[line] = hdfs.get(line, 0) + 1

    print('Number of sessions({}): {}'.format(csv_path, len(hdfs)))
    return hdfs, length

# def generate(name):
#     window_size = 10
#     hdfs = []  # 使用列表来存储所有会话数据，而不是字典
#     length = 0
#     with open('../data/hdfs/' + name, 'r') as f:
#         for ln in f.readlines():
#             ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))  # 将日志中的每个值减 1
#             ln = ln + [-1] * (window_size + 1 - len(ln))  # 填充不足的部分为 -1
#             hdfs.append(ln)  # 将每个会话数据添加到列表中
#             length += 1
#     print('Number of sessions({}): {}'.format(name, length))  # 打印会话总数
#     return hdfs, length



class Predicter():
    def __init__(self, model, options):
        self.options = options
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']

    def predict_unsupervised(self):
        predict_csv = self.options.get('predict_csv_path')
        if not predict_csv:
            predict_csv = os.path.join(
                self.data_dir.rstrip('/\\'), 'demo_input.csv')
        output_csv = self.options.get('predict_output_csv')
        if not output_csv:
            base = os.path.basename(predict_csv)
            output_csv = os.path.join(
                os.path.dirname(os.path.abspath(predict_csv)),
                '..', 'result', 'anomaly_output_for_' + base)
            output_csv = os.path.normpath(output_csv)

        model = self.model.to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        print('predict_csv: {}'.format(predict_csv))
        test_normal_loader, test_normal_length = generate(predict_csv)

        # Test the model
        start_time = time.time()
        anomaly_line_list = []
        with torch.no_grad():
            for line in test_normal_loader.keys():
                for i in tqdm(range(len(line) - self.window_size)):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)

                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        anomaly_line_list.append(i+self.window_size+1)

        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

        df = pd.read_csv(predict_csv)
        filtered_df = df[df['LineId'].isin(anomaly_line_list)]

        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        filtered_df.to_csv(output_csv, index=False)

        print(f"anomaly lines saved to {output_csv}.")
        print(f"anomaly count: {len(filtered_df)}")
        return output_csv, filtered_df

    def predict_supervised(self):
        model = self.model.to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))

