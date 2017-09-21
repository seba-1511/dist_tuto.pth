#!/usr/bin/env python
import os
import torch as th
import torch.distributed as dist
from torch.multiprocessing import Process


def allreduce(send, recv):
    """ Implementation of a ring-reduce. """
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = th.zeros(send.size())
    recv_buff = th.zeros(send.size())
    accum = th.zeros(send.size())
    accum[:] = send[:]
    #th.cuda.synchronize()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send[:]
        send_req.wait()
    #th.cuda.synchronize()
    recv[:] = accum[:]


def run(rank, size):
    """ Distributed function to be implemented later. """
#    t = th.ones(2, 2)
    t = th.rand(2, 2).cuda()
    # for _ in range(10000000):
    for _ in range(4):
        c = t.clone()
        dist.all_reduce(c, dist.reduce_op.SUM)
#        allreduce(t, c)
        t.set_(c)
    print(t)

def init_processes(rank, size, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
#    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.init_process_group(backend, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 1
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
