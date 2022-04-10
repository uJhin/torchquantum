from cmath import pi
from pandas import concat
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
from torchquantum import noise_model
import torchquantum.functional as tqf
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import sys
def binary(input,th = 0.5*0.5*2*3.14159):
    output = input.new(input.size())
    output[input >= th] = 1
    output[input < th] = 0
    return output

def limit(input,high = 1.0,low = 0.0):
    output = input.new(input.size())
    output = input
    output[input >= high] = high
    output[input <= low] = low
    return output
def sign(input,th = 0.5*0.5*2*3.14159):
    output = input.new(input.size())
    output[input >= th] = 1
    output[input < th] = 0
    return output

class RandomDataset(Dataset):
    def __init__(self,sample_num,feature_num):
        normal_data = torch.randn(int(sample_num/2),feature_num,dtype=torch.double)
        binary_data = torch.rand(sample_num-int(sample_num/2),feature_num,dtype=torch.double)
        binary_data = binary(binary_data)
        data01 = (normal_data /16 + 0.25)*2*3.14159
        data02 = (normal_data /16 + 0.75)*2*3.14159
        input_data = torch.concat([data01,data02])
        self.X = limit(input_data,2*3.14159,0)
        weight = torch.rand(feature_num,1,dtype=torch.double)
        self.Y =torch.mm(self.X ,weight)/feature_num
        self.Y = sign(self.Y)
        

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx][0].to(dtype=torch.int64)
        return x,y

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from qiskit.providers.aer.noise import NoiseModel
from qiskit import pulse, QuantumCircuit, execute, Aer
from qiskit.pulse import library
from qiskit.test.mock import FakeQuito, FakeLima, FakeBelem
from qiskit.pulse import transforms
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import IBMQ
from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np
import pdb
pulse_encoding = []
IBMQ.delete_account()
IBMQ.save_account('51a2a5d55d3e1d9683ab4f135fe6fbb84ecf3221765e19adb408699d43c6eaa238265059c3c2955ba59328634ffbd88ba14d5386c947d22eb9a826e40811d626')
IBMQ.load_account()
# provider = IBMQ.load_account()
provider = IBMQ.get_provider(
    hub="ibm-q-research", group="MIT-1", project="main"
)
fake_belem = FakeBelem()
noise_model = NoiseModel.from_backend(fake_belem)
backend = Aer.get_backend('qasm_simulator', noise_model = noise_model)
class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)))

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.rz0(self.q_device, wires=3)
            self.crx0(self.q_device, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
                         parent_graph=self.graph)
            tqf.sx(self.q_device, wires=2, static=self.static_mode,
                   parent_graph=self.graph)
            tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
                     parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

def train(dataflow, model, device, optimizer):
    target_all = []
    output_all = []
    
    for batch_idx, (data, target) in enumerate(dataflow):
        inputs = data.to(device)
        targets = target.to(device)

        outputs = model(inputs)

        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_all.append(targets)
        output_all.append(outputs)

    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return accuracy


def valid_test(dataflow, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataflow):
            inputs = data.to(device)
            targets = target.to(device)
            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    args = parser.parse_args()



    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    best_acc =0
    for epoch in range(1, n_epochs +1):
        # train
        # train
        acc = train(train_data, model, device, optimizer)
        acc = valid_test(test_data, model, device)

        if best_acc<acc:
            best_acc = acc
        print(f'Epoch {epoch}: current acc = {acc},best acc = {best_acc}')

        scheduler.step()

# -*- coding: utf-8 -*-
"""VQPnew.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CZ7HGiMZzaPL29ROUpdn7B35UHysQ01f
"""
if __name__ == '__main__': 
    pdb.set_trace()
    train_db = RandomDataset(100,16)
    train_data = DataLoader(train_db, batch_size=10, shuffle=True)
    test_db = RandomDataset(100,16)
    test_data = DataLoader(test_db, batch_size=10, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    main()