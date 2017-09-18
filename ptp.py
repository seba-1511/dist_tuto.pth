#!/usr/bin/env pybon

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def gather(tensor, rank, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather_recv(tensor_list, tensor, group)
    else:
        dist.gather_send(tensor, root, group)

def run(rank, size):
        """ Simple point-to-point communication. """
        print(dist.get_world_size())
        tensor = torch.ones(1)
        tensor_list = [torch.zeros(1) for _ in range(size)]
        dist.gather(tensor, dst=0, gather_list=tensor_list, group=0)

        print('Rank ', rank, ' has data ', sum(tensor_list)[0])

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
