import os
import numpy as np
import argparse
import pprint
import os.path as pth
import shutil
import random
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_wandb(args):
    # Initialize wandb
    if args.use_wandb:
        import wandb

        wandb.init(
            project="K_QA",
            config=vars(args),
            sync_tensorboard=True,
            name=args.model_name,
            entity="ksikka",
            # tags=args.tags.split(","),
        )
    else:
        wandb = None
    return wandb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, default="model", help="Description")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--result_dir", type=str, default="results/")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--num_validation_samples", type=int, default=500)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--knowledge_method",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="To try different methods for knowledge induction",
    )
    parser.add_argument("--save", dest="save", action="store_true", default=False)
    parser.add_argument(
        "--use_wandb", dest="use_wandb", action="store_true", default=False
    )
    args = parser.parse_args()
    args.model_name += f"_batch_size:{args.batch_size}_lr:{args.lr}"
    if args.knowledge_method:
        args.model_name += f"_knowledge_method:{args.knowledge_method}"
    pprint.pprint(vars(args))
    return args


def create_chkp_result_dirs(checkpoint_dir, result_dir, log_dir, args):
    if pth.isdir(checkpoint_dir):
        print("\t%s already exists. Deleting..." % checkpoint_dir)
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(pth.join(args.logdir, args.model_name))
    if pth.isdir(result_dir):
        print("\t%s already exists. Deleting..." % result_dir)
        shutil.rmtree(result_dir)
    if pth.isdir(log_dir):
        print("\t%s already exists. Deleting..." % log_dir)
        shutil.rmtree(log_dir)

    print("\tCreating dir %s" % checkpoint_dir)
    os.makedirs(checkpoint_dir)
    print("\tCreating dir %s" % result_dir)
    os.makedirs(result_dir)
    print("\tCreating dir %s" % log_dir)
    os.makedirs(log_dir)
