import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from mnist.data import mnist
import numpy as np
import json
import subprocess
import argparse
import sys


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-N', type=int, default=10, help="client number")
parser.add_argument('-dataset', type=str, default='mnist', help="dataset")  # mnist or femnist


def construct_mnist(download=True, client_number=10, data_root='data'):
    print("Construct mnist data ...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    print(data_root)
    test_set = mnist(data_root, False, transform, download=download)
    test_loader = DataLoader(test_set, len(test_set))
    images, labels = [data for data in test_loader][0]
    torch.save((images, labels), os.path.join(data_root, 'mnist', 'processed', 'test-normalize.pt'))
    print("Total Data Info: {} test set: mean={}, std={}".format(len(test_set), images.mean(), images.std()))

    train_set = mnist(data_root, True, transform, download=download)
    train_loader = DataLoader(train_set, len(train_set))
    images, labels = [data for data in train_loader][0]
    torch.save((images, labels), os.path.join(data_root, 'mnist', 'processed','training-normalize.pt'))
    print("Total Data Info: {} train set: mean={}, std={}".format(len(train_set), images.mean(), images.std()))

    data_per_client = len(train_set) // client_number
    index_label_list = []
    for index, label in enumerate(labels):
        index_label_list.append((index, label.item()))
    index_label_list.sort(key=lambda x: x[1])
    data_indicator = torch.tensor([index for index, _ in index_label_list])\
        .reshape(2*client_number, data_per_client//2)
    shuffle_order = np.arange(2*client_number)
    np.random.shuffle(shuffle_order)
    data_indicator = data_indicator[shuffle_order].reshape(client_number, data_per_client)

    clients_folder = os.path.join(data_root, 'mnist', 'processed', 'clients_%s' % client_number)
    os.makedirs(clients_folder, exist_ok=True)
    clients_info_path = os.path.join(clients_folder, "clients.npy")
    np.save(clients_info_path, np.arange(0, client_number))
    print("Client Data Info:")
    for index, indicator in enumerate(data_indicator):
        print("Client %d: %d samples, %s" % (index, len(indicator), [(i,(labels[indicator]==i).sum().item()) for i in range(10)]))
        file_path = os.path.join(clients_folder, "%d.pt" % index)
        torch.save((images[indicator], labels[indicator]), file_path)
    return


def construct_femnist(download=True, client_number=10, data_root='data', percent=0.1):
    data_dir = os.path.join(data_root, 'femnist')
    if download == True:
        def runProcess(script, cwd):
            p = subprocess.Popen(script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
            while (True):
                retcode = p.poll()
                line = p.stdout.readline()
                yield line
                if retcode is not None:
                    break

        script = "./preprocess.sh -s niid --sf %.2f -k 100 -t sample" % percent
        print("generating femnist ...")
        for line in runProcess(script.split(), data_dir):
            print('\r', line.decode().strip('\n'), end='')

    femnist_raw_dir = os.path.join(data_dir, 'data')
    femnist_processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(femnist_processed_dir, exist_ok=True)

    # load test data
    users = []
    num_samples = []
    user_data = []
    for i in range(35):
        f_test = open(os.path.join(femnist_raw_dir, 'test', 'all_data_%d_niid_1_keep_100_test_9.json' % i))
        data = json.load(f_test)
        users += [data['users']]
        num_samples += [data['num_samples']]
        user_data += [data['user_data']]

    # merge test data and save
    test = []  # [(data, label)]
    users[0] = users[0][1:]
    for i in range(0, 35):
        for uid in users[i]:
            test_data = torch.tensor(user_data[i][uid]['x'])
            test_label = torch.tensor(user_data[i][uid]['y'])
            test.append((test_data, test_label))
    test_femnist = list(test[0])
    for i in range(1, len(test)):
        test_femnist[0] = torch.cat((test_femnist[0], test[i][0]), 0)
        test_femnist[1] = torch.cat((test_femnist[1], test[i][1]), 0)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((test_femnist[0].mean(),), (test_femnist[0].std(),))]
    )
    test_femnist[0] = transform(test_femnist[0].numpy()).reshape((-1, 1, 28, 28))
    test_femnist = tuple(test_femnist)
    print("Total Data Info: {} test set: mean={}, std={}".format(len(test_femnist[0]), test_femnist[0].mean(),
                                                                 test_femnist[0].std()))
    torch.save(test_femnist, os.path.join(femnist_processed_dir, "test-normalize.pt"))

    # load original train dataset
    users = []
    num_samples = []
    user_data = {}
    uid = set()
    for i in range(35):
        f_train = open(os.path.join(femnist_raw_dir, 'train', 'all_data_%d_niid_1_keep_100_train_9.json' % i))
        data = json.load(f_train)
        for u in data['users']:
            uid.add(u)
        users += data['users']
        num_samples += data['num_samples']
        user_data.update(data['user_data'])

    # merge train data
    merge_data = {}
    for i in range(client_number):
        rand_set = set(np.random.choice(users, int(len(uid) / client_number), replace=False))
        users = list(set(users) - rand_set)
        merge_data[i] = {'x': [], 'y': []}
        for u in rand_set:
            merge_data[i]['x'] += user_data[u]['x']
            merge_data[i]['y'] += user_data[u]['y']
    noniid_train = []  # [(datas, labels), () ..., ()]
    for uid in range(client_number):
        data = torch.tensor(merge_data[uid]['x'])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((data.mean(),), (data.std(),))])
        data = transform(data.numpy()).reshape((-1, 1, 28, 28))
        label = torch.tensor(merge_data[uid]['y'])
        noniid_train.append((data, label))
    all_train = torch.cat([x[0] for x in noniid_train])
    print("Total Data Info: {} train set: mean={}, std={}".format(len(all_train), all_train.mean(), all_train.std()))

    # split train data and save
    print("Client Data Info:")
    clients_data_dir = os.path.join(femnist_processed_dir, 'clients_%d' % client_number)
    os.makedirs(clients_data_dir, exist_ok=True)
    samples_count = []
    for i in range(client_number):
        sample_stat = sorted([(label, (noniid_train[i][1] == label).sum().item()) for label in range(62)], key=lambda x:x[1], reverse=True)
        print("Client %d: %d samples, %s" % (i, len(noniid_train[i][0]), sample_stat))
        samples_count.append(len(noniid_train[i][0]))
        torch.save(noniid_train[i], os.path.join(clients_data_dir, '%d.pt' % i))
    samples_count = np.array(samples_count)
    print("Sample Distribution: mean={}, std={}, max={}, min={}".format(
        samples_count.mean(), samples_count.std(), samples_count.max(), samples_count.min()))
    clients = np.array(list(range(client_number)))
    np.save(os.path.join(clients_data_dir, "clients.npy"), clients)


def view_image(images, labels):
    assert len(images) == 16 and len(labels) == 16
    for i in range(16):
        plt.subplot(4, 4, 1 + i)
        label = labels[i].item()
        title = u"label: %d" % label
        plt.title(title)
        image = images[i].reshape(28, 28)
        # image = np.fliplr(images[i])
        # image = np.rot90(image)
        plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    client_number = args.N

    data_root = os.getcwd()
    print("*Distribute data among %d clients" % client_number)
    if dataset == "mnist":
        construct_mnist(client_number=client_number, data_root=data_root)
    elif dataset == 'femnist':
        construct_femnist(client_number=client_number, data_root=data_root)
    else:
        print("Dataset should be mnist or femnist!")

    # images, labels = torch.load(os.path.join(data_root, 'mnist', 'processed', 'clients_10', '1.pt'))
    # view_image(images[:16], labels[:16])
