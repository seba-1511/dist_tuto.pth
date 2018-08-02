"""
Distributed Learning using Pytorch's torch.distributed.launcher and
torch.nn.parallel.distributed_c10d on FfDL.
"""

import time
import argparse
import sys
import os
import threading
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.distributed.c10d

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

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
        return F.log_softmax(x, dim=1)


def partition_dataset(batch_size, world_size):
    """ Partitioning MNIST """
    vision_data = os.environ.get("DATA_DIR") + "/data"
    dataset = datasets.MNIST(
        vision_data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))

    bsz = int(batch_size / float(world_size))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    return dataloader, bsz

def average_gradients(model, world_size, pg):
    """ Gradient averaging. """
    for param in model.parameters():
        torch.distributed.c10d.all_reduce(param.grad.data, pg)
        param.grad.data /= world_size


def run(rank, world_rank, world_size, group, batch_size, is_gpu):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    size = os.environ.get("WORLD_SIZE")
    result_dir = os.environ.get("RESULT_DIR") + "/saved_model"
    train_set, bsz = partition_dataset(batch_size, world_size)
    # For GPU use
    if is_gpu:
        # device = torch.device("cuda:{}".format(0))
        # model = Net().to(device)
        model = Net().cuda()
    else:
        model = Net()
        model = model
    model = torch.nn.parallel._DistributedDataParallelC10d(model, group)
#    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            # For GPU use
            if is_gpu:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            if not (size == 1):
                average_gradients(model, world_size, group)
            optimizer.step()
        print('Process ', world_rank,
              ', epoch ', epoch, ': ',
              epoch_loss / num_batches)
    torch.save(model.state_dict(), result_dir)

# Change 'backend' to appropriate backend identifier
def init_processes(local_rank, world_rank, world_size, fn, batch_size, shared_file, is_gpu, backend):
    """ Initialize the distributed environment. """
    print("World Rank: " + str(world_rank) + "  Local Rank: " + str(local_rank)  + " connected")
    pg = torch.distributed.c10d.ProcessGroupGloo(shared_file, world_rank, world_size)
    pg.Options.timeout = 300.0 * world_size
    print("GROUP CREATED")
    fn(local_rank, world_rank, world_size, pg, batch_size, is_gpu)

def local_process(target, args):
    return Process(target=target, args=args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Specify the batch size to be used in training')
    args = parser.parse_args()

    batch_size = args.batch_size
    # Default batch size is set to 1024. When using a large numbers of learners, a larger batch
    # size is sometimes necessary to see speed improvements.
    if batch_size is None:
        batch_size = 1024
    else:
        batch_size = int(batch_size)

    start_time = time.time()
    num_gpus = int(float(os.environ.get("GPU_COUNT")))
    if num_gpus == 0:
        world_size = int(os.environ.get("NUM_LEARNERS"))
    else:
        world_size = num_gpus * int(os.environ.get("NUM_LEARNERS"))
    data_dir = "/job/" + os.environ.get("TRAINING_ID")
    processes = []

    start_time = time.time()
    shared_file = torch.distributed.c10d.FileStore(data_dir)
    processes = []

    print("SHARED FILE PATH: " + data_dir, " WORLD_SIZE: " + str(world_size))
    world_rank = int(os.environ.get("LEARNER_ID")) - 1

    if num_gpus == 0:
        args = (0, world_rank, world_size, run, batch_size, shared_file, True, 'gloo')
        p = local_process(init_processes, args)
        p.start()
        processes.append(p)
    else:
        print("Opening processes")
        for local_rank in range(0, num_gpus):
            args = (local_rank, world_rank, world_size, run, batch_size, shared_file, True, 'gloo')
            p = local_process(init_processes, args)
            print("Process Created")
            p.start()
            processes.append(p)
            print("Process Added")

    for p in processes:
        print("Waiting on Process")
        p.join()

    print("COMPLETION TIME: " + str(time.time() - start_time))

    if int(os.environ.get("LEARNER_ID")) != 1:
        while True:
            time.sleep(1000000)
