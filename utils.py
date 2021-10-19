from torch.utils.data import DataLoader
from torch import nn
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from torch.nn import init

import torch
import os
import math
import sys
import multiprocessing
import numpy as np
import json


def test(model: nn.Module, test_data: DataLoader, device: torch.device):
    total, correct = 0, 0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    with torch.no_grad():
        test_result = []
        for (imgs, labs) in test_data:
            imgs, labs = imgs.to(device), labs.to(device)
            if model.__class__.__name__ in ["LR"]:
                imgs = imgs.view(imgs.size(0), -1)
            outs = model(imgs)
            loss = criterion(outs, labs)
            _, predicted = torch.max(outs.data, 1)
            loss_list.append(loss.item())
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
            test_result.extend(labs[labs==predicted].tolist())
    return correct / total, np.array(loss_list).mean()


def test_model(q: multiprocessing.Queue, S: int, T: int, device: torch.device, Model, params: str,
                test_loader: DataLoader, result_folder):
    print("Test Process: S=%d, T=%d" % (S, T))
    result = []
    result2 = []
    model = Model().to(device)
    for s in range(S):
        result.append([])
        result2.append([])
        for i in range(T):
            agg_model_path = q.get()
            print("test model: ", agg_model_path.format("agg_theta"))
            model.load_state_dict(torch.load(agg_model_path.format("agg_theta")))
            acc, loss = test(model, test_loader, device)
            result[-1].append(acc)
            result2[-1].append(loss)
            print("[Test]:s=%d,i=%d,acc=%.5f,loss=%.5f" % (s, i, result[-1][-1], result2[-1][-1]))
            for name in ['agg_theta', 'agg_grad', 'agg_loss']:
                if i > 0 and os.path.exists(agg_model_path.format(name)):
                    pass
                    os.remove(agg_model_path.format(name))  # delete aggregated model
    with open(os.path.join(result_folder, "{}.acc.json".format(params)), 'w') as f:
        json.dump(result, f)
    with open(os.path.join(result_folder, "{}.loss.json".format(params)), 'w') as f:
        json.dump(result2, f)
    print("[Test Result]: acc={}, loss={}".format(np.mean(result, axis=0), np.mean(result2, axis=0)))
    print("Finish Test!")


def cal_laplace_optimal(N, p, G, mu, _lambda, xi_1, Y0, Gamma, d, epsilon, b_choices, T_choices):
    gamma = 2 * _lambda / mu
    C1 = (4 / mu**2) * (2 * N * G**2) / (N-1)
    C2 = (1/mu**2) * (32 * p * xi_1**2) / (N * d**2) * (N / epsilon**2)
    C3 = gamma * Y0 + (4 / mu**2) * (2 * _lambda * Gamma - 2 * G**2 / (N-1))

    print("-"*50)
    print("[Laplace Optimal(epsilon=%.1f)] C1=%.5f, C2=%.5f, C3=%.5f, gamma=%.5f" % (epsilon, C1, C2, C3, gamma))

    # optimize T and b
    opt_b, opt_T, opt_Tl, opt_U = None, None, None, sys.maxsize
    for i, b in enumerate([1, N]):
        T_original = math.sqrt(gamma ** 2 + (C1 / b + C3) / (C2 * b)) - gamma

        T = math.floor(T_original)
        U_T_b = (C1 / b + C2 * b * T ** 2 + C3) / (T + gamma)
        noise = (b * T / N) * (2 * xi_1 *  N / d) / epsilon
        if U_T_b < opt_U:
            opt_b = b
            opt_T = T
            opt_Tl = float(b)*T/N
            opt_U = U_T_b
        print("Solution %d: b=%d, T=%d, T_l=%d, noise=%.5f, U=%.5f" % (i + 1, b, T, b * T / N, noise, U_T_b))

    print("*Optimal: epsilon={}, b={}, T={}".format(epsilon, opt_b, opt_T))
    joint = {'opt_b': opt_b, 'opt_T': opt_T, 'opt_Tl': opt_Tl, 'opt_U': opt_U}

    # fix b and optimize T
    b_list, T_list, Tl_list, U_list = [], [], [], []
    for b in b_choices:
        A1 = (4 / mu ** 2) * (2 * (N - b) * G ** 2 / (N - 1) / b + 2 * _lambda * Gamma)
        A2 = 32 * p * b * xi_1 ** 2 / mu ** 2 / d ** 2 / epsilon ** 2
        T_original = math.sqrt(gamma ** 2 + (A1 + gamma * Y0) / A2) - gamma
        T = math.floor(T_original)
        U_T_b = (A1 + A2 * T ** 2 + gamma * Y0) / (T + gamma)
        b_list.append((b))
        T_list.append((T))
        Tl_list.append(float(b) * T / N)
        U_list.append(U_T_b)
    fix_b = {'b_list': b_list, 'Tl_list': Tl_list,'T_list':T_list, 'U_list':U_list}

    # fix T and optimize b
    b_list, T_list, Tl_list, U_list = [], [], [], []
    for T in T_choices:
        B1 = (4 / mu ** 2) * (G ** 2 / (T + gamma)) * (2 * N / (N - 1))
        B2 = (1 / mu ** 2) * (1 / (T + gamma)) * (32 * p * T ** 2 * xi_1 ** 2 / d ** 2 / epsilon ** 2)
        B3 = gamma * Y0 / (T + gamma) + 4 / mu ** 2 / (T + gamma) * (2 * _lambda * Gamma - 2 * G ** 2 / (N - 1))
        b_orginal = math.ceil(math.sqrt(B1 / B2))
        b = math.ceil(b_orginal)
        U_T_b = 1 / b * B1 + b * B2 + B3
        b_list.append((b))
        T_list.append((T))
        Tl_list.append(float(b) * T / N)
        U_list.append(U_T_b)
    fix_T = {'b_list': b_list, 'Tl_list': Tl_list, 'T_list': T_list, 'U_list': U_list}

    return {'joint': joint, 'fix_b': fix_b, 'fix_T': fix_T}


def cal_gaussian_optimal(N, p, G, mu, _lambda, xi_2, Y0, Gamma, d, epsilon, Lambda, q, delta, b_choices, max_T=500):
    gamma = 2 * _lambda / mu
    E1 = (8 * G**2 / mu**2) * N / (N-1)

    print("-" * 50)
    print("[Gaussian Optimal(epsilon=%.1f)] E1=%.5f, E2=T-b-related, E3=T-b-related, gamma=%.5f" % (epsilon, E1, gamma))

    b_list, T_list, Tl_list, U_list, sigma_list = [], [], [], [], []
    T = max_T
    di = d/N
    # other
    for i, b in enumerate(b_choices):

        sigma = compute_noise(1, q, epsilon, T * b / N * q, 0.00001, 0.1)
        sigma2 = sigma * xi_2 / (di * q)

        E2 = (4 / mu**2) * (-2 * G**2 / (N-1) + Lambda**2 / q / d + 2 * _lambda * Gamma) + gamma * Y0 - \
             4 / mu**2 * (gamma * p / b / T * sigma2**2)
        E3 = 4 / mu ** 2 * (p / b / T * sigma2**2)
        U_T_b = E1 / b / (T + gamma) + E2 / (T + gamma) + E3
        b_list.append((b))
        T_list.append((T))
        Tl_list.append(b * T / N)
        U_list.append(U_T_b)
        sigma_list.append(sigma)
    fix_b = {'b_list': b_list, 'Tl_list': Tl_list, 'T_list': T_list, 'U_list': U_list, 'sigma_list': sigma_list}

    # optimal
    b = N
    sigma = compute_noise(1, q, epsilon, T * b / N * q, 0.00001, 0.1)
    sigma2 = sigma * xi_2 / (di * q)

    E2 = (4 / mu ** 2) * (-2 * G ** 2 / (N - 1) + Lambda ** 2 / q / d + 2 * _lambda * Gamma) + gamma * Y0 - \
         4 / mu ** 2 * (gamma * p / b / T * sigma2 ** 2)
    E3 = 4 / mu ** 2 * (p / b / T * sigma2 ** 2)
    U_T_b_1 = E1 / b / (T + gamma) + E2 / (T + gamma) + E3

    print("Solution %d-1: b=%d, T=%d, U=%.5f, noise=%.5f" % (0, b, T, U_T_b_1, sigma))
    opt_b = N
    opt_T = max_T
    opt_U = U_T_b_1
    opt_Tl = b * T / N
    opt_sigma = sigma

    T = 0; b = N
    E2_ = (4 / mu ** 2) * (-2 * G ** 2 / (N - 1) + Lambda ** 2 / q / d + 2 * _lambda * Gamma)/gamma + Y0
    U_T_b_2 = E1 / b / (T + gamma) + E2_ / (T + gamma)
    print("Solution %d-2: b=%d, T=%d, U=%.5f, noise=%.5f" % (0, b, T, U_T_b_2, 0))

    if U_T_b_2 < U_T_b_1:
        opt_b = N
        opt_T = 0
        opt_U = U_T_b_2
        opt_Tl = 0

    joint = {'opt_b': opt_b, 'opt_T': opt_T, 'opt_Tl': opt_Tl, 'opt_U': opt_U, 'opt_sigma': opt_sigma}
    print("*Optimal: epsilon={}, b={}, T={}".format(epsilon, opt_b, opt_T))
    print(joint)

    return {'joint': joint, 'fix_b': fix_b, 'fix_T': {}}


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data, 1)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


if __name__ == "__main__":
    print(compute_noise(100, 1, 0.1, 10, 0.00001, 0.1))