from torch.utils.data import DataLoader

import multiprocessing as mp
import argparse
import math
import json
import numpy as np

from datasets import *
from utils import test_model
from Average import FedAvg_L, FedAvg_G
from time import time


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-dp', type=str, default="laplace", help="DP mechanism")  # laplace or gaussian
parser.add_argument('-dataset', type=str, default="mnist", help="dataset")  # mnist or femnist
parser.add_argument('-N', type=int, default=10, help="client number")  # 10, 20, 30, 40, 50
parser.add_argument('-q', type=float, default=1.0, help="local dataset sampling rate")  # 0~1
parser.add_argument('-lr', type=float, default=0.05, help="learning rate")
parser.add_argument('-S', type=int, default=1, help="smooth time")
parser.add_argument('-T', type=int, default=1000, help="total updates")  # global iterations
parser.add_argument('-E', type=int, default=1, help="local updates")  # local iterations
parser.add_argument('-b', type=int, default=10, help="ECP size, b randomly selected clients")  # at most N
parser.add_argument('-mu', type=float, default=0.01, help="weight decay")
parser.add_argument('-C', type=float, default=10000.0, help="gradient clip")
parser.add_argument('-xi', type=float, default=30.0, help="gradient bound")
parser.add_argument('-sigma', type=float, default=2, help="gaussian noise")
parser.add_argument('-epsilon', type=float, default=5000000, help="privacy budget")
parser.add_argument('-device_train', type=str, default='cuda:2')
parser.add_argument('-device_test', type=str, default='cuda:3')
parser.add_argument('-percent', type=float, default=1.0, help="data size")

args = parser.parse_args()
lr = args.lr
S = args.S
T = args.T
E = args.E
b = args.b
mu = args.mu
C = args.C
xi = args.xi
N = args.N
epsilon = args.epsilon
sigma = args.sigma
dp = args.dp
q = args.q
dataset = args.dataset
percent = args.percent

if dataset == 'mnist':
    from models import LR as MODEL
else:
    from models import CNNFemnist as MODEL

test_batch_size = 600

data_root = os.path.join(os.getcwd(), 'data', dataset, 'processed')
clients_data_dir = os.path.join(data_root, 'clients_%d' % N)
model_dir = os.path.join(os.getcwd(), 'temp_model', "{}_{}_{}".format(dataset, dp, N))
result_dir = os.path.join(os.getcwd(), 'result', dataset, 'clients_%d' % N)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

mark = "{}_dp_train_lr={},T={},E={},N={},b={},C={},mu={},epsilon={},q={}".format(dp, lr, T, E, N, b, C, mu, epsilon, q)
print(mark)

# fix random seed
# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed_all(seed)

test_set = test_dataset(os.path.join(data_root, 'test-normalize.pt'))
test_loader = DataLoader(test_set, test_batch_size, shuffle=False)

device_train = torch.device(args.device_train)
device_test = torch.device(args.device_test)


def recv(q_param: mp.Queue, q_res: mp.Queue, S, T):
    for s in range(S):
        for e in range(T):
            n, total_weight = q_param.get()
            model_para = None
            for i in range(n):
                weight, param_path = q_param.get()
                if i == 0:
                    model_para = torch.load(param_path.format("local_theta"))
                    for _, v in model_para.items():
                        v = v.float()
                        v *= weight / total_weight
                else:
                    model = torch.load(param_path.format("local_theta"))
                    for k, v in model_para.items():
                        v = v.float()
                        v += (weight / total_weight) * model[k]
                os.remove(param_path.format("local_theta"))
            agg_model_path = os.path.join(model_dir, "{}_i={}_s={}".format(mark, e+1, s) + ".{}.pth")
            print("save: ", agg_model_path.format('agg_theta'))
            torch.save(model_para, agg_model_path.format('agg_theta'))
            q_res.put(agg_model_path)
        if T == 0:
            print("No aggeregation!")
        else:
            print("Finish Aggregation!")


if __name__ == '__main__':

    T_g = T / E
    T_l = (b * T_g) / N

    all_clients = np.load(os.path.join(clients_data_dir, 'clients.npy'))  # load id list of all clients

    # Use sub-process to test model
    test_T = math.ceil(T_g) + 1
    if b == 0 or T == 0:
        test_T = 1
    q_agg_param = mp.Queue()
    test_process = mp.Process(target=test_model, args=(
        q_agg_param, S, test_T, device_test, MODEL, mark, test_loader, result_dir))
    test_process.start()

    # Use sub-process to aggregate parameters of participating clients and save into files
    q_param = mp.Queue()
    q_result = mp.Queue()
    agg_T = math.ceil(T_g)
    if b == 0 or T == 0:
        agg_T = 0
    aggregate_process = mp.Process(target=recv, args=(q_param, q_result, S, agg_T))
    aggregate_process.start()

    print("Training Process: S=%d, T=%d" % (S, T))
    loss_list = []
    for s in range(S):
        since = time()
        model = MODEL().to(device_train)
        state = model.state_dict()
        total_params = sum(p.numel() for p in model.parameters())
        print('Total Params(p): {}'.format(total_params))

        if dp == 'laplace':
            clip_norm = 1
            Avg = FedAvg_L(MODEL, device_train, E, all_clients, train_dataset, q_param, clients_data_dir, C,
                           T_l=T_l, eplison=epsilon, xi=xi, clip_norm = clip_norm, mu=mu,
                           mark=mark,lr=lr, q=q, model_dir=model_dir, percent=percent)
        else:
            clip_norm = 2
            Avg = FedAvg_G(MODEL, device_train, E, all_clients, train_dataset, q_param, clients_data_dir, C,
                           sigma=sigma, eplison=epsilon, delta=0.00001, xi=xi, clip_norm=clip_norm, mu=mu,
                           mark=mark, lr=lr, q=q, model_dir=model_dir, percent=percent)

        initial_model_param_path = os.path.join(model_dir, "initial_model_s={}.agg_theta.pth".format(s))
        if os.path.exists(initial_model_param_path):
            print("initial model params already exists!")
        else:
            print("create initial model params!")
            torch.save(state, initial_model_param_path)
        q_agg_param.put(initial_model_param_path)

        loss_list.append([])

        # test once although no iterations
        if b == 0 or T == 0:
            print("Warning: no clients participate!")
            continue

        for i in range(math.ceil(T_g)):
            start = time()
            clients = np.arange(N)
            np.random.shuffle(clients)
            if dataset == 'mnist':
                Avg.lr = lr / math.sqrt(i + 1)
            else:
                Avg.lr = lr * 0.9999 ** (int(i))
            loss = Avg.global_update(state, clients[:b])
            loss_list[-1].append(np.mean(loss, axis=0)[0])   # choose the first loss, that is avg loss
            print("[Training]:s=%d,i=%d,local_loss=%s" % (s, i + 1, np.mean(loss, axis=0)))

            # load aggregate params for the next global iteration
            agg_model_path = q_result.get()
            state = torch.load(agg_model_path.format('agg_theta'))
            q_agg_param.put(agg_model_path)
        print("[Training Result-{}]: global_loss={},total_time={}".format(s, loss_list[-1], (time() - since)))
    with open(os.path.join(result_dir, "{}.loss_train.json".format(mark)), 'w') as f:
        json.dump(loss_list, f)
    print("Finish Train!")
