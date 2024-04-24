import math
import wandb


def get_run_parameters():
    p = {
        "model_name": "WangsNet",
        "lr_vec": [0.1, 0.01, 0.001],
        "epochs_vec": [5, 30, 5],
        "learn_vec_lr": False,
        "pic_width": 32,
        "n_samples": 100,
        "cr": 2,
        "save_img": False,
        "n_samples": 20,
        "batch_size": 64,
        "lr": 0.1, # TODO in the paper it's 0.01
        "epochs": 3,
        "optimizer": "adam",
        "weight_decay": 5e-7,
        "num_workers": 4
        # FROM PAPER (section 2.3):
        # keep probability of droput layer = 0.9
        # training step = 300,000
    }
    
    p["img_dim"] = p["pic_width"] * p["pic_width"]
    p["n_masks"] = math.floor(p["img_dim"] / p["cr"])

    if p["learn_vec_lr"]:
        p["epochs"] = sum(p["epochs_vec"])
        p["cum_epochs"] = [sum(p["epochs_vec"][:i + 1]) for i in range(len(p["epochs_vec"]))]

    return p


def load_config_parameters(p):

    wandb.init()
    config = wandb.config

    p["batch_size"] = config.batch_size
    p["pic_width"] = config.pic_width
    p["z_dim"] = config.z_dim
    p["weight_decay"] = config.weight_decay
    p["TV_beta"] = config.TV_beta
    p["cr"] = config.cr

    p["img_dim"] = p["pic_width"] * p["pic_width"]
    p["n_masks"] = math.floor(p["img_dim"] / p["cr"])

    if p["learn_vec_lr"]:
        p["epochs"] = sum(p["epochs_vec"])
        p["cum_epochs"] = [sum(p["epochs_vec"][:i + 1]) for i in range(len(p["epochs_vec"]))]

    return p

