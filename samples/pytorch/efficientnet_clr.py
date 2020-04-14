import argparse
import time

import mlconfig
import torch
from torch import distributed, nn

from .utils import distributed_is_initialized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-parallel', action='store_true')

    # distributed
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    return parser.parse_args()


def init_process(backend, init_method, world_size, rank):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )


def main():    
    args = parse_args()
    train(args.config, 
          args.backend, args.init_method, 
          args.world_size, args.rank, args.no_cuda,
          args.resume, args.data_parallel)


def train(cfg_path, 
          backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=1, rank=0, 
          no_cuda=False, resume=None, data_parallel=False, epoch_cb=None
          ):
    t_start = time.time()
    torch.backends.cudnn.benchmark = True
    cfg = mlconfig.load(cfg_path)
    print(cfg)

    if world_size > 1:
        init_process(backend, init_method, world_size, rank)

    device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

    model = cfg.model()
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        if data_parallel:
            model = nn.DataParallel(model)
        model.to(device)

    optimizer = cfg.optimizer(model.parameters())
    scheduler = cfg.scheduler(optimizer)
    train_loader = cfg.dataset(train=True)
    valid_loader = cfg.dataset(train=False)

    trainer = cfg.trainer(model, optimizer, train_loader, valid_loader, scheduler, device)

    if resume is not None:
        trainer.resume(resume)

    trainer.fit(epoch_cb)
    t_end = time.time() - t_start
    print("Total training time: {:.1f}".format(t_end))


if __name__ == "__main__":
    main()
    
