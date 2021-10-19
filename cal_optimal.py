from utils import *
import argparse
import re

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-dp', type=str, default="laplace", help="DP mechanism")  # laplace or gaussian
parser.add_argument('-dataset', type=str, default="mnist", help="dataset")  # mnist or femnist
parser.add_argument('-N', type=int, default=10, help="client number")  # 10, 20, 30, 40, 50
parser.add_argument('-p', type=int, default=7840, help="model parameters") #  7840 or 110526
parser.add_argument('-d', type=int, default=60000, help="total size of train set") #  60000 or 70147
parser.add_argument('-C', type=int, default=300, help="clip")

args = parser.parse_args()
dp = args.dp
dataset = args.dataset
p = args.p
d = args.d
N = args.N
C = args.C

result_dir = os.path.join(os.getcwd(), 'estimate', dataset, 'clients_%d' % N, 'result')
hyper_params_file_setting_list = []
pattern = '{}_estimate\.hyper_params\.json'.format(dp)
for file_name in os.listdir(result_dir):
    rs = re.match(pattern, file_name)
    if rs is not None:
        hyper_params_file_setting_list.append(file_name)

# settings
epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9, 10]
b_choices = [5, 10, 20, 30] if dp == 'laplace' else [1, 5, 10, 20, 30]
T_choices = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 500, 550]
delta = 0.00001
max_T = 100
q = 0.01

for file_name in hyper_params_file_setting_list:
    print('\n' + "="*100 + '\n')
    print("Optimal results for '{}'".format(file_name))
    with open(os.path.join(result_dir, file_name)) as f:
        hyper_params = json.load(f)
    print("hyper params: ", hyper_params)

    _lambda = hyper_params['lambda']
    _mu = hyper_params['mu']
    xi = min(hyper_params['xi_1'], C) if dp == 'laplace' else min(hyper_params['xi_2'], C)
    gamma = 2 * _lambda / _mu
    G = math.sqrt(hyper_params['G_square'])
    Y0 = hyper_params['Y0']
    Gamma = hyper_params['Gamma']
    Lambda = math.sqrt(hyper_params['Lambda_square'])

    opt_dict = {}
    for i, epsilon in enumerate(epsilon_list):
        b_choices_fixed = []
        for b in b_choices:
            if b <= N:
                b_choices_fixed.append(b)
        if dp == 'laplace':
            rs = cal_laplace_optimal(N, p, G, _mu, _lambda, xi, Y0, Gamma, d, epsilon,
                                     b_choices=b_choices_fixed, T_choices=T_choices)
        else:
            rs = cal_gaussian_optimal(N, p, G, _mu, _lambda, xi, Y0, Gamma, d, epsilon, Lambda, q, delta,
                                      b_choices=b_choices_fixed, max_T=max_T)
        opt_dict['epsilon={}'.format(epsilon)] = rs
    marker = "{}_optimal.json".format(dp)
    print("\nsave into file '{}'".format(marker))
    with open(os.path.join(result_dir, marker), 'w') as f:
        json.dump(opt_dict, f)

print("End!")

