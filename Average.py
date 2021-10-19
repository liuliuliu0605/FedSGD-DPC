import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datetime import datetime
import numpy as np
import os
import sys
from utils import weigth_init


# base
class FedAvg_mp():
    def __init__(self, model, device, E, clients, dataset, queue_param, root, C, clip_norm=1,
                 mark=None, lr=0.1, initial_state=None, mu=0.01, model_dir=None, q=1.0, percent=1.0):
        self.model = model
        self.device = device
        self.mu = mu
        self.E = E
        self.C = C
        self.clip_norm = clip_norm
        self.clients = clients
        self.lr = lr
        self.queue_param = queue_param
        self.root = root
        self.k = 0  # after k global iterations
        self.mark = mark if mark is not None else datetime.timestamp(datetime.now())
        self.dataloader = []
        self.numbers = []
        self.q = q
        print("Dataset sampling rate: {}".format(q))
        for name in self.clients:
            client_data = dataset(self.root, name, percent)
            batch_size = int(client_data.__len__() * q)
            print("initial client {}: batch_size={}".format(name, batch_size))
            self.dataloader.append(DataLoader(client_data, batch_size, shuffle=True))
            self.numbers.append(len(client_data))
        print("total data: {}".format(sum(self.numbers)))
        self.numbers = np.array(self.numbers)
        self.theta_0 = torch.cat([v.reshape(-1) for _, v in initial_state.items()]).to(self.device) if initial_state is not None else None
        self.model_dir = os.path.join(os.getcwd(), 'temp_model') if model_dir is None else model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _client_update(self, id):
        """Update the model on client"""
        model = self.model().to(self.device)
        model.load_state_dict(self.state)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), self.lr)
        dataloader = self.dataloader[id]
        total_loss = []
        for e in range(self.E):
            total_loss.append(0)
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                total_loss[-1] += loss.item()
                for k, v in model.named_parameters():
                    # print(v.grad.norm(2), end='\t')
                    v.grad /= max(1, v.grad.norm(2) / self.C)
                    # print(v.grad.norm(2))
                optimizer.step()
                break
        model_path = "./temp_model/{}_epoch{}_id{}.pth.temp".format(self.mark, self.k, id)
        torch.save(model.state_dict(), model_path)
        self.q.put((self.numbers[id], model_path))
        self.losses.append(total_loss)

    def global_update(self, state, ECP):
        self.state = state
        self.queue_param.put((len(ECP), sum(self.numbers[ECP])))  # (client_size, total_weight) to aggregate
        self.losses = []  # [total_loss_client_i_after_E_epochs ...]
        ECP.sort()
        for id in ECP:
            self._client_update(id)
        self.k += 1
        return self.losses


# estimate params
class FedAvg_estimate(FedAvg_mp):

    def __init__(self, model, device, E, clients, dataset, queue_param, root, C,
                 clip_norm=1, mark=None, lr=0.1, initial_state=None, mu=0.01, model_dir=None, q=1.0, percent=1.0, random=False):
        super().__init__(model, device, E, clients, dataset, queue_param, root, C, clip_norm=clip_norm,
                 mark=mark, lr=lr, mu=mu, model_dir=model_dir, q=q, percent=percent, initial_state=initial_state)
        self.random = random

    def _estimate_params(self, last_theta, last_grad, last_loss, cur_theta, cur_grad, cur_loss):
        theta_delta = -(cur_theta - last_theta)
        grad_delta = -(cur_grad - last_grad)

        division = theta_delta.norm(2).item()
        assert division != 0

        _lambda = grad_delta.norm(2).item()/division if division != 0 else 0  # λ-smooth
        mu = _lambda

        xi_1 = last_grad.norm(1).item()  # l1-norm of gradients boundary
        xi_2 = last_grad.norm(2).item()  # l2-norm of gradients boundary
        G_square = last_grad.norm(2).item() ** 2
        Lambda_square = last_grad

        return {'lambda':_lambda, 'mu': mu, 'G_square': [G_square], 'xi_1': xi_1, 'xi_2': xi_2, 'Lambda_square': [Lambda_square]}

    def _handle(self, estimated_params, rs):
        estimated_params['mu'] = min(rs['mu'], estimated_params.get('mu', sys.maxsize))
        estimated_params['lambda'] = max(rs['lambda'], estimated_params.get('lambda', 0))
        estimated_params['xi_1'] = max(rs['xi_1'], estimated_params.get('xi_1', 0))
        estimated_params['xi_2'] = max(rs['xi_2'], estimated_params.get('xi_2', 0))
        estimated_params['G_square'] = estimated_params.get('G_square', []) + rs['G_square']
        estimated_params['Lambda_square'] = estimated_params.get('Lambda_square', []) + rs['Lambda_square']

    def _getOptimal(self, id):
        local_optimal_model_path = os.path.join(self.model_dir, "{}_i={}_id={}".format(self.mark, 1, id) + ".{}.pth")
        print(local_optimal_model_path.format('local_theta'))
        if os.path.exists(local_optimal_model_path.format('local_theta')):
            local_optimal_model = self.model().to(self.device)
            local_optimal_model.load_state_dict(torch.load(local_optimal_model_path.format('local_theta')))
            local_optimal_theta = torch.cat([w.reshape(-1) for w in local_optimal_model.parameters()])
        else:
            local_optimal_theta = None
        local_optimal_loss = np.array(torch.load(local_optimal_model_path.format('local_loss')))[-1] if \
            os.path.exists(local_optimal_model_path.format('local_loss')) else None
        return local_optimal_theta, local_optimal_loss

    def _client_update(self, id):
        """Update the model on client"""
        model = self.model().to(self.device)
        model.load_state_dict(self.state)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(model.parameters(), self.lr, weight_decay=self.mu)
        dataloader = self.dataloader[id]
        loss_list = []

        last_theta, last_grad, last_loss = None, None, None

        estimated_params = {}
        for e in range(self.E):  # local iterations
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                cur_theta = torch.cat([w.reshape(-1) for w in model.parameters()])
                if model.__class__.__name__ in ["LR"]:
                    data = data.view(data.size(0), -1)
                output = model(data)
                loss = criterion(output, target)
                loss = loss.mean()

                # calculate loss
                loss.backward()
                cur_loss = loss.item()
                loss_list.append(cur_loss)

                # clip the gradient
                for k, v in model.named_parameters():
                    v.grad /= max(1, v.grad.norm(self.clip_norm) / self.C)  # clip gradient, Laplace 1 norm, Gaussian 2 norm
                cur_grad = torch.cat([w.grad.reshape(-1) for w in model.parameters()])

                if last_theta is not None and last_grad is not None and last_loss is not None:
                    rs = self._estimate_params(last_theta, last_grad, last_loss, cur_theta, cur_grad, cur_loss)
                    self._handle(estimated_params, rs)

                last_theta, last_grad, last_loss = cur_theta, cur_grad, cur_loss

                # update model parameters
                optimizer.step()
                if self.random and e % 2 == 1:
                    last_theta, last_grad, last_loss = None, None, None
                    model.apply(weigth_init)
                break

        local_model_path = os.path.join(self.model_dir, "{}_i={}_id={}".format(self.mark, self.k + 1, id) + ".{}.pth")
        if not self.random:
            torch.save(model.state_dict(), local_model_path.format('local_theta'))  # save local model parameters
            torch.save(cur_grad, local_model_path.format('local_grad'))  # save local gradients
            torch.save(loss_list, local_model_path.format('local_loss'))  # save local loss
        estimated_params['G_square'] = np.max(estimated_params['G_square'])
        estimated_params['Lambda_square'] = torch.stack(estimated_params['Lambda_square'], 0).\
                                                var(0).norm(1).item() / self.E
        self.losses.append(loss_list)

        if not self.random:
            theta = torch.cat([w.reshape(-1) for w in model.parameters()])
            loss = loss_list[-1]
        else:
            theta, loss = self._getOptimal(id)
            if theta is None and loss is None:
                print("ERROR: please estimate with random=False first!")
                exit(1)
        estimated_params['Y0'] = (self.theta_0 - theta).norm(2).item() ** 2
        estimated_params['loss'] = loss

        print("id={},loss={}...{}".format(id, loss_list[:5], loss_list[-5:]))
        print("id={},params={}".format(id, estimated_params))

        self.queue_param.put((self.numbers[id], local_model_path, estimated_params))


# laplace
class FedAvg_L(FedAvg_mp):
    def __init__(self, model, device, E, clients, dataset, queue_param, root, C,
                 T_l, eplison, xi, clip_norm = 1, mu=0.1,
                 mark=None, lr=0.1, q=1, model_dir=None, percent=1.0):
        super().__init__(model, device, E, clients, dataset, queue_param, root, C, clip_norm = clip_norm,
                         mark=mark, lr=lr, mu=mu, q=q, model_dir=model_dir, percent=percent)
        self.eplison = eplison
        self.T_l = T_l
        self.xi = xi

    def _client_update(self, id):
        """Update the model on client"""
        model = self.model().to(self.device)
        model.load_state_dict(self.state)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), self.lr, weight_decay=self.mu)
        dataloader = self.dataloader[id]
        total_loss = []

        # set clip as xi
        layers = len(list(model.named_parameters()))
        C = self.C / layers
        self.xi = self.C

        l1_sensitivity = 2 * self.xi / self.numbers[id]
        beta = l1_sensitivity * self.T_l / self.eplison  # calculate noise
        for e in range(self.E):
            total_loss.append(0)
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                data = data.view(data.size(0), -1)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                total_loss[-1] += loss.item()
                for k, v in model.named_parameters():
                    v.grad /= max(1, v.grad.norm(self.clip_norm) / C)
                    if self.eplison > 0 :
                        noise = torch.from_numpy(np.random.laplace(0, scale=beta, size=v.shape)).to(self.device)
                        v.grad += noise
                    pass
                optimizer.step()
        if self.k == 0:
            print("client:{} with noise:{}".format(id, beta))
        local_model_path = os.path.join(self.model_dir, "{}_i={}_id={}".format(self.mark, self.k + 1, id) + ".{}.pth")
        torch.save(model.state_dict(), local_model_path.format('local_theta'))
        self.queue_param.put((self.numbers[id], local_model_path))  # (weight_client_i, param_path_client_i)
        self.losses.append(total_loss)


# gaussian
class FedAvg_G(FedAvg_mp):
    def __init__(self, model, device, E, clients, dataset, queue_param, root, C,
                 sigma, eplison, delta, xi, clip_norm = 2, mu=0.1,
                 mark=None, lr=0.1, q=0.1, model_dir=None, percent=1.0):
        super().__init__(model, device, E, clients, dataset, queue_param, root, C, clip_norm = clip_norm,
                         mark=mark, lr=lr, mu=mu, q=q, model_dir=model_dir, percent=percent)
        self.eplison = eplison
        self.delta = delta
        self.sigma = sigma
        self.xi = xi

    def _client_update(self, id):
        """Update the model on client"""
        model = self.model().to(self.device)
        model.load_state_dict(self.state)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), self.lr, weight_decay=self.mu)
        dataloader = self.dataloader[id]
        total_loss = []

        layers = len(list(model.named_parameters()))
        C = self.C / layers
        self.xi = self.C
        L = self.q * self.numbers[id]

        for e in range(self.E):
            total_loss.append(0)
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                data = data.view(data.size(0), -1)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                total_loss[-1] += loss.item()

                for k, v in model.named_parameters():
                    v.grad /= max(1, v.grad.norm(self.clip_norm) / C)
                    if self.eplison > 0:
                        noise = (torch.from_numpy(np.random.normal(0, scale=self.sigma*self.C/L, size=v.shape))).to(self.device)
                        v.grad += noise
                    pass
                optimizer.step()
                break
        if self.k == 0:
            print("client:{} with noise:{}".format(id, self.sigma*self.C/L), self.C, L)
        local_model_path = os.path.join(self.model_dir, "{}_i={}_id={}".format(self.mark, self.k + 1, id) + ".{}.pth")
        torch.save(model.state_dict(), local_model_path.format('local_theta'))
        self.queue_param.put((self.numbers[id], local_model_path))  # (weight_client_i, param_path_client_i)
        self.losses.append(total_loss)
