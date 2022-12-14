import random
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as T
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import torchvision.models as models


from utils import get_config, setup_seed, unwrap_model
from model import VisionTransformer

device = torch.device("cuda:0")

def neighbor(item, item_list):
    if len(item_list) == 1:
        return item_list[0]
    idx = item_list.index(item)
    if idx == 0:
        c = [0, 1]
    elif idx == len(item_list) - 1:
        c = [0, -1]
    else:
        c = [1, 0, -1]
    return item_list[idx + random.choice(c)]

class AutoFormerSpace:
    def __init__(self, search_embed_dim, search_depth, search_num_heads, search_num_ratio):
        self.search_embed_dim = search_embed_dim
        self.search_depth = search_depth
        self.search_num_heads = search_num_heads
        self.search_num_ratio = search_num_ratio
        self.depth = max(search_depth)

        self.config = None

    def typology(self):
        if self.config is None:
            return self.max()
        config = deepcopy(self.config)
        # config["embed_dim"] = neighbor(config["embed_dim"], self.search_embed_dim)
        # config["depth"] = neighbor(config["depth"], self.search_depth)
        for i in range(6, self.depth):
            config["num_heads"][i] = neighbor(config["num_heads"][i], self.search_num_heads)
            config["mlp_ratio"][i] = neighbor(config["mlp_ratio"][i], self.search_num_ratio)
        self.config = config
        return config

    def random(self):
        # config = {
        #     "embed_dim": random.choice(self.search_embed_dim),
        #     "depth": random.choice(self.search_depth),
        #     "num_heads": [random.choice(self.search_num_heads) for _ in range(self.depth)],
        #     "mlp_ratio": [random.choice(self.search_num_ratio) for _ in range(self.depth)]
        # }
        config = {
            "embed_dim": random.choices(self.search_embed_dim, [0.35, 0.3, 0.35])[0],
            "depth": random.choices(self.search_depth, [0.35, 0.3, 0.35])[0],
            "num_heads": [random.choices(self.search_num_heads)[0] for _ in range(self.depth)],
            "mlp_ratio": [random.choices(self.search_num_ratio, [0.35, 0.3, 0.35])[0] for _ in range(self.depth)]
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

setup_seed(42)
cfg = get_config("config/imagenet-100.yaml", "")
search_space = AutoFormerSpace(cfg.search_space.search_embed_dim, cfg.search_space.search_depth, cfg.search_space.search_num_heads, cfg.search_space.search_num_ratio)

cfg.model.num_classes = cfg.data.num_classes
model = VisionTransformer(**cfg.model).to(device)

# from sam import SAM

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ])
        norm = torch.norm(norm, p=2)
        
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

sam = True
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
optimizer = optim.AdamW(unwrap_model(model).parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
if sam:
    optimizer = SAM(unwrap_model(model).parameters(), optimizer.__class__, rho=0.05, **optimizer.defaults)

for it in range(8):
    inputs, targets = torch.rand(8, 3, 224, 224), torch.randint(0, 100, (8, ))
    batch_size = inputs.size(0)
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    unwrap_model(model).set_sample_config(search_space.random())
    logits = model(inputs)

    if sam:
        # first forward-backward pass
        loss = criterion(logits, targets)  # use this loss for any training statistics
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        criterion(model(inputs), targets).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)

    else:
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(it, loss.item())

exit()

config = search_space.min()
model.set_sample_config(config)
min_params = model.get_params()/1e6
config = search_space.max()
model.set_sample_config(config)
max_params = model.get_params()/1e6

print(min_params, max_params)

# for i in [192, 216, 240]:
#     config["embed_dim"] = i
# for i in [12, 13, 14]:
#     config["depth"] = i
#     model.set_sample_config(config)
#     params = model.get_params()/1e6
#     print(i, params)
# exit()

params = []
for _ in tqdm(range(10000), ncols=80):
    model.set_sample_config(autoformer.random())
    params.append(model.get_params()/1e6)

params = np.array(params)
with plt.style.context('ggplot'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=120)
    ax.axvline(min_params, color="green", label="min")
    ax.axvline(max_params, color="red", label="max")
    n, bins, patches = ax.hist(params, 100, density=True, facecolor='skyblue', alpha=1., label="sample")

    ax.set_xlabel('Params')
    ax.set_ylabel('Probability')
    ax.set_title('Histogram of Params')
    ax.grid(True)
    ax.legend()
    plt.savefig("output.png", dpi=120)
    plt.close()
exit()
# inputs = torch.rand(1, 3, 224, 224).to(device)

# with torch.no_grad():
#     model.set_sample_config(autoformer.min())
#     model(inputs)
#     print(model.get_params())


# IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
# IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

# import cv2
# from PIL import Image
# def cv2_loader(path: str):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     img = cv2.imread(path)
#     height, width, _ = img.shape
#     ratio = 256 / min(height, width)
#     img = cv2.resize(img, (int(width*ratio), int(height*ratio)), interpolation = cv2.INTER_LINEAR)
#     # You may need to convert the color.
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(img)


# train_transform = T.Compose([
#     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
#     T.RandomCrop(224),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
# ])
# valid_transform = T.Compose([
#     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#     # T.Resize(224, interpolation=T.InterpolationMode.NEAREST),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
# ])
# # loader=cv2_loader
# train_set = datasets.ImageFolder("/root/rjchen/data/imagenet/train", transform=train_transform)
# valid_set = datasets.ImageFolder("/root/rjchen/data/imagenet/val", transform=valid_transform)

# dataloader_config = {
#     "batch_size": 512,
#     "num_workers": 8,
#     "drop_last": True,
#     "pin_memory": True,
#     "shuffle": True,
#     "persistent_workers": True
# }
# # train_loader = DataLoader(train_set, sampler = RandomSampler(train_set), **dataloader_config)
# # valid_loader = DataLoader(valid_set, sampler = SequentialSampler(valid_set), **dataloader_config)
# train_loader = DataLoader(train_set, **dataloader_config)
# valid_loader = DataLoader(valid_set, **dataloader_config)

# from tqdm import tqdm
# dataloader = train_loader

# device = torch.device("cuda:0")
# model = models.resnet18(num_classes=1000, weights=None)
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# for epoch in range(5):
#     with tqdm(dataloader, total=len(dataloader), ncols=80, desc=f"Train({epoch})") as t:
#         for inputs, targets in t:
#             inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
#             loss = criterion(model(inputs), targets)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             t.set_postfix_str(f"loss: {loss.item():.3f}")
#         scheduler.step()
