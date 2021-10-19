# Optimizing the Numbers of Queries and Replies in Federated Learning with Differential Privacy
This repository is the source code for the paper: https://arxiv.org/pdf/2107.01895.pdf

## Requirements
To install requirements:
```
pip install -r requirements.txt
```
* Python3
* Pytorch
* Torchvision
* TensorflowPrivacy


## Datasets
We provide two datasets under corresponding folders.
* **MNIST**: 10 different classes, 70,000 handwritten digit images with 28 by 28 gray-scale pixels. 
* **FEMNIST**: 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels 
(with option to make them all 128 by 128 pixels), 3500 users.

For quickly generating datasets, run the following scripts in _data_ folders:

```
python generate_data.py -N 10 -dataset mnist
python generate_data.py -N 10 -dataset femnist
```

### References
```
@misc{title={LEAF: A Benchmark for Federated Settings},
author={Sebastian Caldas and Sai Meher Karthik Duddu and Peter Wu and Tian Li and Jakub Konečný and H. Brendan McMahan and Virginia Smith and Ameet Talwalkar},
year={2018},
eprint={1812.01097},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

## Parameter Estimation
We estimate the problem-specific parameters with two methods: **learning process** and **random initialization**.
* **learning process**: each client achieves its local optimal all by oneself, so total iterations *T* and 
local iterations *E* have the same value (e.g., 200).
The results of estimation are saved in **estimate/$dataset/clients_$N/**.
```
python estimate.py -dp=laplace -dataset=mnist -T=200 -E=200 -N=10 -b=10 -q=1.0
python estimate.py -dp=gaussian -dataset=mnist -T=200 -E=200 -N=10 -b=10 -q=0.01
```

* **random initialization**: each client randomly initializes its model parameters and performs SGD
steps.
The results of estimation are saved in **estimate/$dataset/clients_$N/**.
```
python estimate.py -dp=laplace -dataset=mnist -T=100 -E=100 -N=10 -b=10 -q=1.0 -random
python estimate.py -dp=gaussian -dataset=mnist -T=100 -E=100 -N=10 -b=10 -q=0.01 -random
```

## Optimizing *T* and *b*
Calculate the optimal values of *T* and *b* according to parameters estimated above and 
the optimal results are stored in **estimate/$dataset/clients_$N/result/$dp**
```
python cal_optimal.py -dataset=mnist -dp=laplace -N=10 -p=7840 -d=60000 -C=300
python cal_optimal.py -dataset=mnist -dp=gaussian -N=10 -p=7840 -d=60000 -C=10
```

## Training 
Training with optimal values of *T* and *b*, and other candidates. The results of loss and 
accuracy are in result **result/$dataset/clients_$N**
```
python batch_dp_train.py -dp laplace -dataset mnist -S 10 -q 1.0 -C 300
python batch_dp_train.py -dp gaussian -dataset mnist -S 10 -q 0.01 -C 10
```

## Results
* **log/**: training log. 
* **estimate/$dataset/clients_$N/result/$dp/**: estimated parameters and optimal results.
* **result/$dataset/client_$N/**: training (test) loss and accuracy.

## Code Parameters
* ```-q``` := sampling probability, 1.0 for Laplace mechanism, or 0.01 for Gaussian mechanism
* ```-dataset``` := datasets used, either mnist or femnist.
* ```-p``` := number of model parameters, e.g., 7,840 and 11,0526 for mnist and femnist, respectively.
* ```-d``` := total number of training datasets, e.g. 60,000 for mnist.