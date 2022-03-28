from cmath import pi
from pandas import concat
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import sys
def binary(input,th = 0.5):
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
def sign(input,th = 0.5):
    output = input.new(input.size())
    output[input >= th] = 1
    output[input < th] = -1
    return output

class RandomDataset(Dataset):
    def __init__(self,sample_num,feature_num):
        normal_data = torch.randn(int(sample_num/2),feature_num,dtype=torch.double)
        binary_data = torch.rand(sample_num-int(sample_num/2),feature_num,dtype=torch.double)
        binary_data = binary(binary_data)
        data001 = (normal_data /16 + 0.25)*2*3.14159
        data002 = (normal_data /16 + 0.75)*2*3.14159
        data01=0.8*data001+0.2*data002 + 0.12* data001 + 0.14 * data001
        data02=0.2*data001+0.8*data002 + 0.12*data002 + 0.18 * data002
        input_data = torch.concat([data01,data02])
        self.X = limit(input_data,2*3.14159,0)
        weight = torch.rand(feature_num,1,dtype=torch.double)
        self.Y =torch.mm(self.X ,weight)
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
from qiskit import pulse, QuantumCircuit
from qiskit.pulse import library
from qiskit.test.mock import FakeQuito
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
backend = provider.get_backend('ibmq_jakarta')
class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=1,
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
        backend = provider.get_backend('ibmq_jakarta')
        bsz = x.shape[0]
        x = x.view(bsz, 16)
        circs_pulse = self.encoder.to_qiskit(4,x)
        print(circs_pulse)
        circs_pulse[0].draw()
        circs_pulse = transpile(circs_pulse, backend = backend, basis_gates=['u1', 'u2', 'u3', 'cx'], initial_layout = initial_mapping, optimization_level=2)
        backend = provider.get_backend('ibmq_jakarta')
        for i in range (0, len(circs_pulse)):
                    with pulse.build(backend) as pulse_enco:
                         pulse.call(circs_pulse[i])
                            
                    pulse_encoding.append(pulse_enco)
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
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')

    args = parser.parse_args()



    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    best_acc =0
    for epoch in range(1, n_epochs +1):
        # train
        # train
        acc = train(train_data, model, device, optimizer)
        if best_acc<acc:
            best_acc = acc
        print(f'Epoch {epoch}: current acc = {acc},best acc = {best_acc}')

        scheduler.step()

import math
import qiskit
from qiskit import pulse, QuantumCircuit
from qiskit.pulse import library
from qiskit.test.mock import FakeQuito
from qiskit.pulse import transforms
from qiskit.pulse.transforms import block_to_schedule
from qiskit.pulse import filters
from qiskit.pulse.filters import composite_filter, filter_instructions
from typing import List, Tuple, Iterable, Union, Dict, Callable, Set, Optional, Any
from qiskit.pulse.instructions import Instruction
from qiskit.compiler import assemble, schedule, transpile
import numpy as np
import torch.nn.functional as F
import torch
import time
from torchquantum.datasets import MNIST
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.optimize import LinearConstraint
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from scipy.stats import norm
import pdb
from qiskit.compiler import assemble, schedule
initial_mapping = [5, 3, 4, 1] 
backend = provider.get_backend('ibmq_jakarta')

with pulse.build(backend) as pulse_prog:
        qc = QuantumCircuit(4)
        qc.cx(1, 0)
        qc.rz(-4.1026, 3)
        qc.cx(0,1)
        qc.h(3)
        qc.rx(1.2803, 0)
        qc.ry(0.39487, 1)
        qc.crx(-3.025, 0, 2)
        qc.sx(2)
        qc.cx(3,1)
        qc.measure_all()
        qc = transpile(qc, backend = backend, basis_gates=['u1', 'u2', 'u3', 'cx'], initial_layout = initial_mapping, optimization_level=2)
        print(qc)
        pulse.call(qc)
        print(pulse)
        sched = pulse.Schedule()
        
pulse_prog.draw()

"""Extract amps"""

def is_parametric_pulse(t0, *inst: Union['Schedule', Instruction]):
    inst = t0[1]
    t0 = t0[0]
    if isinstance(inst, pulse.Play) and isinstance(inst.pulse, pulse.ParametricPulse):
        return True
    return False

amps = [play.pulse.amp for _, play in pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions]
print(amps)

for _, play in pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions:
    # print(play.pulse.amp)
    pass
    
    
instructions = pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions

amp_list = list(map(lambda x: x[1].pulse.amp, pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions))
amp_list = np.array([amp_list])
ampa_list = np.angle(np.array([amp_list]))
ampn_list = np.abs(np.array([amp_list]))
amps_list = np.append(ampn_list, ampa_list)
print(amps_list)
rag = np.arange(1,1.1,0.05)
amps_list = [amps_list*x for x in rag]
amps_list = np.array(amps_list)


def get_expectations_from_counts(counts, qubits):
    exps = []
    if isinstance(counts, dict):
        counts = [counts]
    for count in counts:
        ctr_one = [0] * qubits
        total_shots = 0
        for k, v in count.items():
            k = "{0:04b}".format(int(k, 16))
            for qubit in range(qubits):
                if k[qubit] == '1':
                    ctr_one[qubit] += v
            total_shots += v
        prob_one = np.array(ctr_one) / total_shots
        exp = np.flip(-1 * prob_one + 1 * (1 - prob_one))
        exps.append(exp)
    res = np.stack(exps)
    return res

"""BO"""

def acquisition(x_scaled, hyper_param, model, min_Y):  # x_scaled: 1 * dim
    x_scaled = x_scaled.reshape(1, -1)
    if 'LCB' in hyper_param[0]:
        mean, std = model.predict(x_scaled, return_std=True)
        return mean[0] - hyper_param[1] * std[0]
    elif hyper_param[0] == 'EI':
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = (tau - mean) * norm.cdf(tau_scaled) + std * norm.pdf(tau_scaled)
        return -res  # maximize Ei = minimize -EI
    elif hyper_param[0] == 'PI':
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = norm.cdf(tau_scaled)
        return -res
    else:
        raise ValueError("acquisition function is not implemented")

def bayes_opt(func, dim_design, N_sim, N_initial, w_bound, hyper_param, store=False, verbose=True, file_suffix=''):
    '''

    :param func: [functional handle], represents the objective function. objective = func(design)
    :param dim_design: [int], the dimension of the design variable
    :param N_sim: [int], The total number of allowable simulations
    :param N_initial: [int], The number of simulations used to set up the initial dataset
    :param w_bound: [(dim_design, 2) np.array], the i-th row contains the lower bound and upper bound for the i-th variable
    :param hyper_param: the parameter for the acquisition function e.g., ['LCB','0.3'], ['EI'], ['PI']
    :param verbose: [Bool], if it is true, print detailed information in each iteration of Bayesian optimization
    :param file_suffix: [string], file suffix used in storing optimization information
    :return:
    cur_best_w: [(dim_design,) np.array], the best design variable
    cur_best_y: [float], the minimum objective value
    '''

    # initialization: set up the training dataset X, Y.
    print("Begin initializing...")
    N_initial = N_initial
    X = amps_list
    print(X)
    Y = np.zeros((N_initial,))

    # todo: 因为BO没法直接输出complex number，我们需要把实数和虚数部分分成两个部分来训练，再组合成一整个X放回simulator里。
    for i in range(N_initial):
        
        Y[i] = func(X[i, :])
        print("Simulate the %d-th sample... with metric: %.3e" % (i, Y[i])) if verbose else None
    print("Finish initialization with best metric: %.3e" % (np.min(Y)))

    # define several working variables, will be used to store results
    pred_mean = np.zeros(N_sim - N_initial)
    pred_std = np.zeros(N_sim - N_initial)
    acq_list = np.zeros(N_sim - N_initial)

    # Goes into real Bayesian Optimization
    cur_count, cur_best_w, cur_best_y = N_initial, None, 1e10
    while cur_count < N_sim:

        # build gaussian process on the normalized data
        wrk_mean, wrk_std = X.mean(axis=0), X.std(axis=0)
        model = GPR(kernel=ConstantKernel(1, (1e-9, 1e9)) * RBF(1.0, (1e-5, 1e5)), normalize_y=True,
                    n_restarts_optimizer=100)
        model.fit(np.divide(X - wrk_mean, wrk_std, out=np.zeros_like(X-wrk_mean), where=wrk_std!=0), Y)

        # define acquisition function, np.min(Y) is needed in EI and PI, but not LCB
        acq_func = lambda x_scaled: acquisition(x_scaled, hyper_param, model, np.min(Y))

        # optimize the acquisition function independently for N_inner times, select the best one
        N_inner, cur_min, opt = 20, np.inf, None
        for i in range(N_inner):
            w_init = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
                w_bound[:, 0])
            lb = np.divide(w_bound[:, 0] - wrk_mean, wrk_std+1e-8)
            #out=np.zeros_like(w_bound[:, 0]-wrk_mean), where=wrk_std!=0
            ub = np.divide(w_bound[:, 1] - wrk_mean, wrk_std+1e-8)
            #out=np.zeros_like(w_bound[:, 1]-wrk_mean), where=wrk_std!=0
            LC = LinearConstraint(np.eye(dim_design), lb, ub, keep_feasible=False)
            cur_opt = minimize(acq_func, np.divide(w_init - wrk_mean, wrk_std, out=np.zeros_like(w_init - wrk_mean), where=wrk_std!=0), method='COBYLA', constraints=LC,
                               options={'disp': False})
            wrk = acq_func(cur_opt.x)
            if cur_min >= wrk:
                cur_min = wrk
                opt = cur_opt

        # do a clipping to avoid violation of constraints (just in case), and also undo the normalization
        newX = np.clip(opt.x * wrk_std + wrk_mean, w_bound[:, 0], w_bound[:, 1])
        star_time = time.time()
        cur_count += 1
        newY = func(newX)
        end_time = time.time()
        X, Y = np.concatenate((X, newX.reshape(1, -1)), axis=0), np.concatenate((Y, [newY]), axis=0)

        # save and display information
        ind = np.argmin(Y)
        cur_predmean, cur_predstd = model.predict((np.divide(newX - wrk_mean, wrk_std, out=np.zeros_like(newX - wrk_mean), where=wrk_std!=0)).reshape(1, -1), return_std=True)
        cur_acq = acq_func(np.divide(newX - wrk_mean, wrk_std, out=np.zeros_like(newX - wrk_mean), where=wrk_std!=0))
        cur_best_w, cur_best_y = X[ind, :], Y[ind]
        pred_mean[cur_count - N_initial - 1], pred_std[cur_count - N_initial - 1] = cur_predmean, cur_predstd
        acq_list[cur_count - N_initial - 1] = cur_acq
        if store:
            np.save('./result/X_' + file_suffix + '.npy', X)
            np.save('./result/Y_' + file_suffix + '.npy', Y)
            np.save('./result/cur_best_w_' + file_suffix + '.npy', cur_best_w)
            np.save('./result/cur_best_y_' + file_suffix + '.npy', cur_best_y)
            np.save('./result/pred_mean_' + file_suffix + '.npy', pred_mean)
            np.save('./result/pred_std_' + file_suffix + '.npy', pred_std)
            np.save('./result/acq_list_' + file_suffix + '.npy', acq_list)
        if verbose:
            print("-" * 10)
            print("Number of function evaluations: %d" % cur_count)
            print("Optimize acq message: ", opt.message)
            print("Model predict(new sampled X)... mean: %.3e, std:%.3e" % (cur_predmean, cur_predstd))
            print("Acq(new sampled X): %.3e" % cur_acq)
            print("Y(new sampled X): %.3e, simulation time: %.3e" % (newY, end_time - star_time))
            print("Current best design: ", cur_best_w)
            print("Current best function value: %.3e" % cur_best_y)

    return cur_best_w, cur_best_y

def Fucsimulate(cur_best_w):
    mid = int(len(cur_best_w)/2)
    modified_list = ((cur_best_w[:mid])*np.cos(cur_best_w[mid:]) + (cur_best_w[:mid])*np.sin(cur_best_w[mid:])*1j)
    modified_list = np.ndarray.tolist(modified_list)
    backend = provider.get_backend('ibmq_jakarta')
    target_all = []
    output_all = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with pulse.build(backend) as pulse_prog:
            qc = QuantumCircuit(4)
            qc.cx(1, 0)
            qc.rz(-4.1026, 3)
            qc.cx(0,1)
            qc.h(3)
            qc.rx(1.2803, 0)
            qc.ry(0.39487, 1)
            qc.crx(-3.025, 0, 2)
            qc.sx(2)
            qc.cx(3,1)
            qc.measure_all()
            print(qc)
            qc = transpile(qc, backend = backend, basis_gates=['u1', 'u2', 'u3', 'cx'],initial_layout = initial_mapping, optimization_level=2)
            pulse.call(qc)
            print(pulse)
    for inst, amp in zip(pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions, modified_list):
        inst[1].pulse._amp = amp

        #quito_sim = qiskit.providers.aer.PulseSimulator.from_backend(FakeQuito())
    for i in range(0, len(pulse_encoding)):
        pulse_sim = assemble(pulse_prog + pulse_encoding[i], backend=backend, shots=128)
        results = backend.run(pulse_sim).result()
        counts = results.data()['counts']
        result = get_expectations_from_counts(counts, 4)
        result = torch.tensor(result)
        bsz = result.shape[0]
        result = result.reshape(bsz, 2, 2).sum(-1)     
        result = F.log_softmax(result, dim=1)
        output_all.append(result)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_data):
            inputs = data.to(device)
            targets = target.to(device)

            target_all.append(targets)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return 1 - accuracy

def test(cur_best_w):
    modified_list = ((cur_best_w[:int(len(cur_best_w)/2)])*np.cos(cur_best_w[int(len(cur_best_w)/2):]) + (cur_best_w[:int(len(cur_best_w)/2)])*np.sin(cur_best_w[int(len(cur_best_w)/2):])*1j)
    modified_list = np.ndarray.tolist(modified_list)
    target_all = []
    output_all = []


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    with pulse.build(backend) as pulse_prog:
            qc = QuantumCircuit(4)
            qc.cx(1, 0)
            qc.rz(-4.1026, 3)
            qc.cx(0,1)
            qc.h(3)
            qc.rx(1.2803, 0)
            qc.ry(0.39487, 1)
            qc.crx(-3.025, 0, 2)
            qc.sx(2)
            qc.cx(3,1)
            qc.measure_all()
            qc = transpile(qc, backend = backend, basis_gates=['u1', 'u2', 'u3', 'cx'], initial_layout = initial_mapping, optimization_level=2)
            print(qc)
            pulse.call(qc)
            print(pulse)
    for inst, amp in zip(pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions, modified_list):
        inst[1].pulse._amp = amp
    for i in range(0, len(pulse_encoding)):
        pulse_sim = assemble(pulse_prog + pulse_encoding[i], backend=backend, shots=128, meas_level = 2, meas_return = 'single')
        results = backend.run(pulse_sim).result()
        counts = results.data()['counts']
        result = get_expectations_from_counts(counts, 4)
        result = torch.tensor(result)
        bsz = result.shape[0]
        result = result.reshape(bsz, 2, 2).sum(-1)
        result = F.log_softmax(result, dim=1)
        output_all.append(result)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data):
            inputs = data.to(device)
            targets = target.to(device)

            target_all.append(targets)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return 1 - accuracy


if __name__ == '__main__':
    initial_mapping = [5, 3, 4, 1] 
    pdb.set_trace()
    train_db = RandomDataset(20,16)
    train_data = DataLoader(train_db, batch_size=20, shuffle=True)
    test_db = RandomDataset(20,16)
    test_data = DataLoader(test_db, batch_size=20, shuffle=True)



    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    main()
    print(pulse_encoding)
    seed = 0
    np.random.seed(seed)
    # example: minimize x1^2 + x2^2 + x3^2 + ...
    dim_design = int(len(amps_list[0]))
    Mid = int(len(amps_list[0])/2)
    N_total = 15
    N_initial = 3
    bound = np.ones((dim_design, 2)) * np.array([0, 1]) 
    bound[-Mid:] = bound[-Mid:]*360 # -inf < xi < inf

    func = Fucsimulate
    cur_best_w, cur_best_y = bayes_opt(func, dim_design, N_total, N_initial, bound, ['LCB', 0.3],
                                       store=False, verbose=True, file_suffix=str(seed))

    print(cur_best_w)
    print(cur_best_y)
    accuracy = test(cur_best_w)
    print(accuracy)
