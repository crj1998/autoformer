
import os, logging, argparse, random
import wandb
from copy import deepcopy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


from utils import get_logger, get_config, colorstr, setup_seed, unwrap_model, AverageMeter
from model import VisionTransformer
from sam import SAM

class AutoFormerSpace:
    def __init__(self, search_embed_dim, search_depth, search_num_heads, search_num_ratio):
        self.search_embed_dim = search_embed_dim
        self.search_depth = search_depth
        self.search_num_heads = search_num_heads
        self.search_num_ratio = search_num_ratio
        self.depth = max(search_depth)

        self.config = None

    def random(self):
        config = {
            "embed_dim": random.choice(self.search_embed_dim),
            "depth": random.choice(self.search_depth),
            "num_heads": [random.choice(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [random.choice(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return config
    
    def min(self):
        config = {
            "embed_dim": min(self.search_embed_dim),
            "depth": min(self.search_depth),
            "num_heads": [min(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [min(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return config

    def max(self):
        config = {
            "embed_dim": max(self.search_embed_dim),
            "depth": max(self.search_depth),
            "num_heads": [max(self.search_num_heads) for _ in range(self.depth)],
            "mlp_ratio": [max(self.search_num_ratio) for _ in range(self.depth)]
        }
        self.config = config
        return config

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
        dist.barrier()
        reduced_acc = reduce_mean(acc, args.world_size)
        Acc.update(reduced_acc.item(), batch_size)
        Loss.update(loss.item(), batch_size)

    return Loss.item(), Acc.item()

def train(args, model, dataloader, criterion, optimizer, scheduler, mixup_fn=None, search_space=None, teacher_model=None):
    epoch, iters = args.epoch, args.valid_step
    Loss = AverageMeter()
    dataiter = iter(dataloader)
    model.train()
    # lr = scheduler.get_last_lr()[0]
    # lr = optimizer.param_groups[0]['lr']
    for it in range(iters):
        try:
            inputs, targets = next(dataiter)
        except:
            dataiter = iter(dataloader)
            inputs, targets = next(dataiter)

        batch_size = inputs.size(0)
        inputs, targets = inputs.to(args.device, non_blocking=True), targets.to(args.device, non_blocking=True)
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        if args.sample == "random":
            unwrap_model(model).set_sample_config(search_space.random())
        logits = model(inputs)

        if args.sam:
            # first forward-backward pass
            loss = criterion(logits, targets)  # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            criterion(model(inputs), targets).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

        else:
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                loss = F.kl_div(
                    F.log_softmax(logits / args.distill_temp, dim=-1),
                    F.softmax(teacher_logits / args.distill_temp, dim=-1),
                    reduction="batchmean"
                ) * (args.distill_temp ** 2)
            else:
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Loss.update(loss.item(), batch_size)

        if args.local_rank in [-1, 0] and ((it+1) % args.log_interval == 0 or it+1 == iters or it == 0):
            args.logger.info(f"Epoch {epoch:>2d} Iter {it+1:>4d}: loss={Loss.item():6.4f}")
    scheduler.step(epoch)
    return Loss.item()


def main(args):
    # set up logger
    cfg = get_config(args.cfg, args.override)
    level = logging.DEBUG if "dev" in args.out else logging.INFO

    # level = logging.INFO
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1

    logger = get_logger(
        name = f"AutoFormer{args.local_rank}", output = args.out,
        level = level if args.local_rank in [-1, 0] else logging.WARN,
        fmt = "%(asctime)s [%(levelname)s] %(message)s" if args.local_rank in [-1, 0] else "%(asctime)s [%(levelname)s @ %(name)s] %(message)s", 
        rank = args.local_rank
    )
    args.logger = logger
    logger.debug(f"Get logger named {colorstr('AutoFormer')}!")
    logger.debug(f"Distributed available? {colorstr(str(dist.is_available()))}")

    if args.local_rank in [-1, 0] and args.wandb:
        wandb.init(project="autoformer", entity="maze", name=args.out.split("/")[-1], config=cfg.state_dict())
    #setup random seed
    if args.seed is not None and isinstance(args.seed, int):
        setup_seed(args.seed)
        logger.info(f"Setup random seed {colorstr('green', args.seed)}!")
    else:
        logger.info(f"Can not Setup random seed with seed is {colorstr('green', args.seed)}!")
    
    # init dist params
    # args.n_gpu = torch.cuda.device_count()
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
        # IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
        # IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
        # train_transform = T.Compose([
        #     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        #     T.RandomCrop(224),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # ])
        valid_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        trans = cfg.data.trans
        train_transform = create_transform(
            input_size=cfg.data.img_size,
            is_training=True,
            color_jitter=trans.color_jitter,
            auto_augment=trans.aa,
            interpolation=trans.train_interpolation,
            re_prob=trans.reprob,
            re_mode=trans.remode,
            re_count=trans.recount,
        )
        train_set = ImageFolder(os.path.join(cfg.data.datafolder, "train"), transform=train_transform)
        valid_set = ImageFolder(os.path.join(cfg.data.datafolder, "val"), transform=valid_transform)

        logger.info(f"Dataset: {colorstr('green', len(train_set))} samples for train, {colorstr('green', len(valid_set))} sampels for valid!")
    # prepare dataloader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    valid_sampler = SequentialSampler if args.local_rank == -1 else DistributedSampler
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    num_classes = cfg.data.num_classes
    dataloader_config = {
        "batch_size": batch_size//args.world_size,
        "num_workers": num_workers,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True
    }
    train_loader = DataLoader(train_set, sampler=train_sampler(train_set), **dataloader_config)
    valid_loader = DataLoader(valid_set, sampler=valid_sampler(valid_set), **dataloader_config)

    logger.info(f"Dataloader Initialized. Batch num: {colorstr('green', len(train_loader))}, Batch size: {colorstr('green', batch_size)}, Num workers: {colorstr('green', num_workers)}.")

    aug = cfg.data.aug
    mixup_active = aug.mixup > 0 or aug.cutmix > 0. or aug.cutmix_minmax is not None
    mixup_fn = Mixup(
        mixup_alpha=aug.mixup, cutmix_alpha=aug.cutmix, cutmix_minmax=aug.cutmix_minmax,
        prob=aug.mixup_prob, switch_prob=aug.mixup_switch_prob, mode=aug.mixup_mode,
        label_smoothing=aug.smoothing, num_classes=num_classes
    ) if mixup_active else None

    search_space = AutoFormerSpace(cfg.search_space.search_embed_dim, cfg.search_space.search_depth, cfg.search_space.search_num_heads, cfg.search_space.search_num_ratio)
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
            logger.warning(f"Missing keys {msg.missing_keys} in state dict.")
            logger.info(f"Pretrained weights @: {colorstr(str(args.pretrained))} loaded!")

        args.teacher = False
        teacher_model = deepcopy(model) if args.teacher else None
    
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(args.device)
    if args.teacher:
        teacher_model.to(args.device)
    total_epoch = cfg.scheduler.epochs
    args.total_step = len(train_loader) * total_epoch
    args.valid_step = len(train_loader)
    args.sam = cfg.train.sam
    args.sample = cfg.train.sample
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    # optimizer = optim.AdamW(unwrap_model(model).parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    # scheduler = cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_epoch * args.valid_step, num_training_steps=args.total_step
    # )
    if aug.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif aug.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=aug.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_valid = nn.CrossEntropyLoss()
    optimizer = create_optimizer(cfg.optimizer, unwrap_model(model))
    if args.sam:
        # optimizer.__class__(optimizer.param_groups, **optimizer.defaults)
        optimizer = SAM(unwrap_model(model).parameters(), optimizer.__class__, rho=0.05, **optimizer.defaults)
        
    scheduler, _ = create_scheduler(cfg.scheduler, optimizer)

    logger.info(f"Optimizer {colorstr('Adamw')} and Scheduler {colorstr('Cosine')} selected!")

    if args.local_rank != -1:
        # This mode allows running backward on a subgraph of the model, and DDP finds out which parameters are involved in the backward pass by traversing the autograd graph from the model output and marking all unused parameters as ready for reduction. 
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    best_acc = 0.0
    args.global_step = 0
    args.epoch = 0
    #train loop
    model.zero_grad()
    
    for epoch in range(1, total_epoch+1):
        args.epoch = epoch
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        if args.sample in ["random", "max"]:
            unwrap_model(model).set_sample_config(search_space.max())
        elif args.sample == "min":
            unwrap_model(model).set_sample_config(search_space.min())
        else:
            pass
        train_loss = train(args, model, train_loader, criterion, optimizer, scheduler, mixup_fn, search_space, teacher_model)
        if args.sample in ["random", "max"]:
            unwrap_model(model).set_sample_config(search_space.max())
        elif args.sample == "min":
            unwrap_model(model).set_sample_config(search_space.min())
        else:
            pass
        valid_loss, valid_acc = valid(args, model, valid_loader, criterion_valid)
        if args.local_rank in [-1, 0]:
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "best_model.pth"))
            torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "last_model.pth"))
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"LR: {lr:.8f} Train loss: {train_loss:.3f} Valid loss: {valid_loss:.3f} Valid acc: {valid_acc:.2%}")
            if args.wandb:
                wandb.log({
                    "train/loss": round(train_loss, 4),
                    "train/lr": round(lr, 8),
                    "valid/loss": round(valid_loss, 4),
                    "valid/acc": round(valid_acc, 4)
                }, step=epoch)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('AutoFormer training')
    parser.add_argument('--cfg', type=str, required=True, help='a config file')
    parser.add_argument('--out', default='../results/develop', help='directory to output the result')
    parser.add_argument('--pretrained', default=None, help='directory to pretrained model')
    # parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--log_interval', default=64, type=int, help="log print interval")
    parser.add_argument('--wandb', action="store_true", help="use wandb")
    parser.add_argument('--override', default='', type=str, help='overwrite the config, keys are split by space and args split by |, such as train.eval_step=2048|optimizer.lr=0.1')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    main(args)