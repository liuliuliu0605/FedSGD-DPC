from torch.utils.data import DataLoader
from datasets import *
from utils import weigth_init
from Average import FedAvg_estimate as FedAvg
from time import time

import multiprocessing as mp
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('-dp', type=str, default="laplace", help="DP mechanism")  # laplace or gaussian
parser.add_argument('-dataset', type=str, default="mnist", help="dataset")  # mnist or femnist
parser.add_argument('-T', type=int, default=100, help="total updates")  # total iterations
parser.add_argument('-E', type=int, default=100, help="local updates")  # local iterations
parser.add_argument('-N', type=int, default=10, help="client number")  # total clients
parser.add_argument('-b', type=int, default=10, help="b randomly selected clients")  # at most N
parser.add_argument('-q', type=float, default=1.0, help="local dataset sampling rate")  # 0~1
parser.add_argument('-mu', type=float, default=0.0001, help="weight decay")
parser.add_argument('-C', type=float, default=10000.0, help="gradient clip")
parser.add_argument('-lr', type=float, default=0.05, help="learning rate")
parser.add_argument('-device_train', type=str, default='cuda:2')
parser.add_argument('-device_test', type=str, default='cuda:3')
parser.add_argument('-random', action="store_true", help="random")
parser.add_argument('-percent', type=float, default=1.0, help="data size")


test_batch_size = 600

args = parser.parse_args()
dp = args.dp
dataset = args.dataset
T = args.T
E = args.E
N = args.N
b = args.b
q = args.q
mu = args.mu
C = args.C
lr = args.lr
Random = args.random
percent = args.percent
mark = "{}_estimate".format(dp)

if dataset == 'mnist':
    from models import LR as LR
else:
    from models import CNNFemnist as LR

data_root = os.path.join(os.getcwd(), 'data', dataset, 'processed')
clients_data_dir = os.path.join(data_root, 'clients_%d' % N)
estimate_dir = os.path.join(os.getcwd(), 'estimate', dataset, 'clients_%d' % N)
model_dir = os.path.join(estimate_dir, 'model')
result_dir = os.path.join(estimate_dir, 'result')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

test_set = test_dataset(os.path.join(data_root, 'test-normalize.pt'))
test_loader = DataLoader(test_set, test_batch_size, shuffle=False)

device_train = torch.device(args.device_train)
device_test = torch.device(args.device_test)


def aggregate(q_param: mp.Queue, q_result: mp.Queue, global_iterations):
    for i in range(global_iterations):
        client_size, total_weight = q_param.get()
        agg_hyper_params = {}
        max_loss = 0
        for j in range(client_size):  # agg params from all clients
            weight, client_model_path, hyper_params = q_param.get()
            max_loss = max(max_loss, hyper_params['loss'])
            if j == 0:
                for name, value in hyper_params.items():
                    if name in ['xi_1', 'xi_2', 'lambda', 'G_square', 'mu', 'Lambda_square']:
                        agg_hyper_params[name] = value
                    else:
                        agg_hyper_params[name] = weight / total_weight * value
            else:
                for name, value in hyper_params.items():
                    if name in ['xi_1', 'xi_2', 'lambda', 'G_square', 'Lambda_square']:
                        agg_hyper_params[name] = max(agg_hyper_params[name], value)
                    elif name in ['mu']:
                        agg_hyper_params[name] = min(agg_hyper_params[name], value)
                    else:
                        agg_hyper_params[name] += weight / total_weight * value
        agg_hyper_params['Gamma'] = max_loss - agg_hyper_params['loss']
        print("[Estimated Params]: i=%d, params=%s" % (i, agg_hyper_params))
    print("Done!")
    # save estimated parameters in the last global iteration
    if not Random:
        with open(os.path.join(result_dir, "%s.hyper_params_first.json" % mark), 'w') as f:
            json.dump(agg_hyper_params, f)
    else:
        with open(os.path.join(result_dir, "%s.hyper_params.json" % mark), 'w') as f:
            json.dump(agg_hyper_params, f)


if __name__ == '__main__':
    T_g = T // E  # global iteration, number of aggregations by server
    T_l = (b * T_g) // N  # clients participation, number of client response times
    if T_g * E != T:
        print("WARNING! T_g ({}) * E ({}) != T ({})".format(T_g, E, T))
        exit(1)
    if T_l * N != b * T_g:
        print("WARNING! T_l ({}) * N ({}) != b ({}) * T_g ({})".format(T_l, N, b, T_g))
        exit(1)

    # Randomly selected b participating clients and guarantee each client is selected once
    # in N //b global iterations
    all_clients = np.load(os.path.join(clients_data_dir, 'clients.npy'))  # load id list of all clients
    clients = []
    for _ in range(T_l):
        temp_clients = np.arange(N)
        np.random.shuffle(temp_clients)
        clients.append(temp_clients.reshape((-1, b)))
    clients = np.concatenate(clients)

    # Use sub-process to aggregate parameters of participating clients and save into files
    q_param = mp.Queue()
    q_result = mp.Queue()
    aggregate_process = mp.Process(target=aggregate, args=(q_param, q_result, T_g))
    aggregate_process.start()


    since = time()
    model = LR().to(device_train)
    initial_model_param_path = os.path.join(model_dir, "{}_i={}".format(mark, 0) + ".{}.pth")
    if not Random:
        model.apply(weigth_init)
        state = model.state_dict()
        torch.save(state, initial_model_param_path.format('agg_theta'))
    else:
        state = torch.load(initial_model_param_path.format('agg_theta'))

    total_params = sum(p.numel() for p in model.parameters())
    print('Total Params(p): {}'.format(total_params))

    clip_norm = 1 if dp == 'laplace' else 2
    Avg = FedAvg(LR, device_train, E, all_clients, train_dataset, q_param, clients_data_dir, C,
                 mark=mark, lr=lr, mu=mu, initial_state=state, clip_norm=clip_norm,
                 model_dir=model_dir, q=q, percent=percent, random=Random)

    for i in range(T_g):
        start = time()
        loss = Avg.global_update(state, clients[i])
