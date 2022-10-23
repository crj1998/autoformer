
import os, sys, logging, argparse, random

import csv
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



from utils import get_config, colorstr, setup_seed, unwrap_model, AverageMeter
from model import VisionTransformer
from searcher import EvolutionSearcher

def get_logger(name, level=logging.INFO, fmt="%(asctime)s [%(levelname)s @ %(name)s] %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # output to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    return logger


class AutoFormerSpace:
    def __init__(self, search_embed_dim, search_depth, search_num_heads, search_num_ratio):
        self.search_embed_dim = search_embed_dim
        self.search_depth = search_depth
        self.search_num_heads = search_num_heads
        self.search_num_ratio = search_num_ratio
        self.depth = max(search_depth)

        self.config = None

    def space_to_dict(self):
        d = {"embed_dim": self.search_embed_dim, "depth": self.search_depth}
        for i in range(self.depth):
            d[f"num_heads_{i}"] = self.search_num_heads
            d[f"mlp_ratio_{i}"] = self.search_num_ratio
        return d
    
    def to_dict(self, config):
        config = dict(config)
        d = {"embed_dim": config['embed_dim'], "depth": config['depth']}
        for i in range(self.depth):
            d[f"num_heads_{i}"] = config['num_heads'][i]
            d[f"mlp_ratio_{i}"] = config['mlp_ratio'][i]
        return d

    def from_dict(self, config):
        config = dict(config)
        d = {"embed_dim": config['embed_dim'], "depth": config['depth'], 'mlp_ratio': [], 'num_heads': []}
        for i in range(self.depth):
            d['mlp_ratio'].append(config.pop(f'mlp_ratio_{i}'))
            d['num_heads'].append(config.pop(f'num_heads_{i}'))
        return d

    def random(self):
        config = {
            "embed_dim": random.choice(self.search_embed_dim),
            "depth": random.choice(self.search_depth),
            "num_heads": [random.choice(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [random.choice(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return config
    
    def min(self, full_dict=False):
        config = {
            "embed_dim": min(self.search_embed_dim),
            "depth": min(self.search_depth),
            "num_heads": [min(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [min(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return self.to_dict(config) if full_dict else config

    def max(self, full_dict=False):
        config = {
            "embed_dim": max(self.search_embed_dim),
            "depth": max(self.search_depth),
            "num_heads": [max(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [max(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return self.to_dict(config) if full_dict else config

@contextmanager
def torch_distributed_zero_first(rank):
    """ Decorator to make all processes in distributed training wait for each local_master to do something. """
    if rank not in [-1, 0]:
        dist.barrier(device_ids=[rank])
    yield
    if rank == 0:
        dist.barrier(device_ids=[0])

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

@torch.no_grad()
def valid(args, model, dataloader, criterion):
    Acc = AverageMeter()
    Loss = AverageMeter()
    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        acc = (logits.argmax(dim=-1)==targets).sum()/batch_size
        if args.local_rank != -1:
            dist.barrier()
            reduced_acc = reduce_mean(acc, args.world_size)
            Acc.update(reduced_acc.item(), batch_size)
        else:
            Acc.update(acc.item(), batch_size)
        Loss.update(loss.item(), batch_size)

    return Loss.item(), Acc.item()



def main(args):
    # set up logger
    cfg = get_config(args.cfg, args.override)
    level = logging.INFO

    # level = logging.INFO
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    logger = get_logger(
        name = f"AutoFormer{args.local_rank}", level = level if args.local_rank in [-1, 0] else logging.WARN,
        fmt = "%(asctime)s [%(levelname)s] %(message)s" if args.local_rank in [-1, 0] else "%(asctime)s [%(levelname)s @ %(name)s] %(message)s"
    )
    args.logger = logger
    logger.debug(f"Get logger named {colorstr('AutoFormer')}!")
    logger.debug(f"Distributed available? {colorstr(str(dist.is_available()))}")

    #setup random seed
    if args.seed is not None and isinstance(args.seed, int):
        setup_seed(args.seed)
        logger.info(f"Setup random seed {colorstr('green', args.seed)}!")
    else:
        logger.info(f"Can not Setup random seed with seed is {colorstr('green', args.seed)}!")
    
    # init dist params
    if args.local_rank == -1:
        args.world_size = 1
        device = torch.device('cuda')
    else:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        # args.local_rank = dist.get_rank()
        # torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        assert dist.is_initialized(), f"Distributed initialization failed!"

    # set device
    args.device = device
    logger.debug(f"Current device: {device}")

    # make dataset
    with torch_distributed_zero_first(args.local_rank):
        transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        dataset = ImageFolder(os.path.join(cfg.data.datafolder, "val"), transform=transform)

        logger.info(f"Dataset: {colorstr('green', len(dataset))} sampels for valid!")
    # prepare dataloader
    sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset, shuffle=False, drop_last=False)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    num_classes = cfg.data.num_classes
    dataloader_config = {
        "batch_size": 250, # batch_size//args.world_size
        "num_workers": num_workers,
        # "drop_last": False,
        "pin_memory": True,
        "persistent_workers": True
    }
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_config)

    logger.info(f"Dataloader Initialized. Batch size: {colorstr('green', batch_size)}, Num workers: {colorstr('green', num_workers)}.")

    with torch_distributed_zero_first(args.local_rank):
        # build model
        cfg.model.num_classes = num_classes
        model = VisionTransformer(**cfg.model)
        logger.info(f"Model: {colorstr('ViT')}. Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

        # load from pre-trained, before DistributedDataParallel constructor
        # pretrained is str and it exists and is file.
        if isinstance(args.pretrained, str) and os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            state_dict = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v.clone() for k, v in state_dict.items()}

            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) != 0: 
                logger.warning(f"Missing keys {msg.missing_keys} in state dict.")
            logger.info(f"Pretrained weights @: {colorstr(str(args.pretrained))} loaded!")
        else:
            logger.error(f"Pretrained weights @: {colorstr(str(args.pretrained))} not loaded!")
    
    model.to(args.device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.local_rank != -1:
        # This mode allows running backward on a subgraph of the model, and DDP finds out which parameters are involved in the backward pass by traversing the autograd graph from the model output and marking all unused parameters as ready for reduction. 
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    search_space = AutoFormerSpace(cfg.search_space.search_embed_dim, cfg.search_space.search_depth, cfg.search_space.search_num_heads, cfg.search_space.search_num_ratio)

    def eval_func(config):
        config = search_space.from_dict(config)

        unwrap_model(model).set_sample_config(config)
        params = unwrap_model(model).get_params()
        valid_loss, valid_acc = valid(args, model, dataloader, criterion)
        logger.info(f"Subnet: {colorstr('blue', config)}. Params: {colorstr('green', str(round(params/1e6, 2))+' M'):>8s} Acc: {colorstr('green', str(round(valid_acc*100, 2)) + '%'):>8s}")
        return {'metric': valid_acc, 'params': round(params/1e6, 2), 'acc': round(valid_acc*100, 2)}

    name = args.name
    if name == 'evolution':
        evolution_iters = 20
        population_num = 50
    elif name == 'random':
        evolution_iters = 0
        population_num = 1000
    else:
        evolution_iters = 0
        population_num = 32
    searcher = EvolutionSearcher(eval_func, search_space.space_to_dict(), evolution_iters=evolution_iters, population_num=population_num, optimize='max')
    topk = searcher.run([search_space.min(True), search_space.max(True)])

    with torch_distributed_zero_first(args.local_rank):
        if args.local_rank in [-1, 0]:
            for k, v in topk.items():
                print(search_space.from_dict(k), v)
            with open(os.path.join(args.out, f'subnet_{args.name}.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                depth = search_space.depth
                writer.writerow(['embed_dim', 'depth'] + [f'num_heads_{i}' for i in range(depth)] + [f'mlp_ratio_{i}' for i in range(depth)] + ['Params', 'Accs'])
                for subnet, metric in searcher._cache.items():
                    writer.writerow([
                        subnet['embed_dim'], subnet['depth']] + 
                        [subnet[f'num_heads_{i}'] for i in range(depth)] + 
                        [subnet[f'mlp_ratio_{i}'] for i in range(depth)] + 
                        [metric['params'], metric['acc']]
                    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('AutoFormer training')
    parser.add_argument('--cfg', type=str, required=True, help='a config file')
    parser.add_argument('--out', default='../results/search', required=True, help='directory to output the result')
    parser.add_argument('--pretrained', default=None, help='directory to pretrained model')
    # parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--name', type=str, required=True, help='search type')
    parser.add_argument('--override', default='', type=str, help='overwrite the config, keys are split by space and args split by |, such as train.eval_step=2048|optimizer.lr=0.1')
    args = parser.parse_args()
    if args.pretrained is None:
        args.pretrained = os.path.join(args.out, "last_model.pth")
    main(args)

"""
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 28100 search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random/best_model.pth
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 28100 search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random/best_model.pth
CUDA_VISIBLE_DEVICES=0 python search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random/best_model.pth

"""