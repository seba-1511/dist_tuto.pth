"""
Distributed Learning using distributed.init_process_group and the environment
variable initialization method. Remote machines are accessed using Paramiko
and processes are started on the remote machines to run the job and communicate
back to the master process.
"""

import time
import sys
import os
import paramiko
import threading
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
#    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

# Change 'backend' to appropriate backend identifier
def init_processes(rank, size, fn, m_address, m_port, backend):
    """ Initialize the distributed environment. """
    print("Rank " + str(rank) + " connected")
    os.environ['MASTER_ADDR'] = m_address
    os.environ['MASTER_PORT'] = m_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def start_process(creds, rank, size, master_creds):
    address, user, pass_key, path_to_model = creds
    m_address, m_port, m_backend = master_creds
    print("Enter Remote Process")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(address, username=user, password=pass_key)
    stdin, stdout, stderr = client.exec_command(
        "python -u " + path_to_model + " " + str(rank) + " " + str(size) +
        " " + m_address + " " + m_port + " " + m_backend)
    print("End Remote Process")
    return (stdin, stdout, stderr, client)

def start_thread(stream):
    thread = threading.Thread(target=monitor_process, args=[stream])
    thread.daemon = True
    thread.start()

def monitor_process(stream):
    x = stream.readline()
    while not (x == ""):
        print(x)
        x = stream.readline()

def local_process(target, args):
    return Process(target=target, args=args)

def remote_bot(rank, size, m_address, m_port, m_backend):
    print("REMOTE BOT READY")
    init_processes(rank, size, run, m_address, m_port, m_backend)

def parse_file(file_path):
    file = open(file_path, 'r')
    cur_line = file.readline()

    while not cur_line == "":
        items = cur_line.split(" ")

        host = items[0]
        rank = items[0].split("-")[-1]
        slots = items[1].split("=")[1]


if __name__ == "__main__":

    if len(sys.argv) > 2:
        rank = int(sys.argv[1])
        size = int(sys.argv[2])
        remote_bot(rank, size, sys.argv[3], sys.argv[4], sys.argv[5])
    else:

        start_time = time.time()
        path_to_model = "dist_tuto.pth/train_distributed.py"
        password = ""
        # The address, port, and backend communication for the master process
        master_creds = ['169.45.95.130', '29500', 'gloo']
        # List of four tuples of the address, user, password, and path to model for each
        # process. The rank of the process corresponds to its index in this list.
        # If local processes need to be added, add "local" instead of a four tuple
        process_creds = [("169.45.95.130", "abutler", password, path_to_model),
                         ("169.45.95.130", "abutler", password, path_to_model)]

        remote_clients = []
        local_processes = []
        size = len(process_creds)

        for process_rank in range(0, len(process_creds)):
            p_cred = process_creds[process_rank]
            if p_cred == "local":
                m_addr, m_port, backend = master_creds
                p = local_process(init_processes, (process_rank, size, run, m_addr, m_port, backend))
                p.start()
                local_processes.append(p)
            else:
                (stdin, stdout, stderr, client) = start_process(p_cred, process_rank, size, master_creds)
                start_thread(stdout)
                start_thread(stderr)
                remote_clients.append(client)

        while remote_clients:
            client_num = 0
            while client_num < len(remote_clients):
                client = remote_clients[client_num]

                if not client.get_transport().is_active():
                    remote_clients.pop(client_num)
                    client.close()
                else:
                    client_num += 1
            print("Current Time: ", time.time() - start_time)
            time.sleep(30)
