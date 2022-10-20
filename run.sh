# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 train.py --cfg config/imagenet-100.yaml --out ../results/in100_max
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 train.py --cfg config/imagenet-100.yaml --out ../results/in100_random_sam --wandb

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 28100 search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random/best_model.pth
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 28100 search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random/best_model.pth --name random
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 --master_port 28100 search.py --cfg config/imagenet-100.yaml --out ../results/in100_min/ --name random
CUDA_VISIBLE_DEVICES=0 python search.py --cfg config/imagenet-100.yaml --pretrained ../results/in100_random_sam/best_model.pth --name dev