import time

import torch
import torch.optim as optim

import NN_architecture
from DataFunctions import get_data
from Training import train_epoch
from Testing import test_net
from LogFunctions import print_and_log_message
from LogFunctions import print_training_messages
from OutputHandler import save_orig_img, save_all_run_numerical_outputs


def train(params, log_path, folder_path, wb_flag):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = build_network(params["pic_width"], params["n_masks"], params["model_name"], device)
    optimizer = build_optimizer(network, params["optimizer"], params["lr"], params["weight_decay"])
    train_loader, test_loader = get_data(
        params["pic_width"],
        params["n_masks"],
        params["batch_size"],
        params["n_samples"]
    )
    numerical_outputs = {
        "train_loss": [],
        "test_loss": [],
        "train_psnr": [],
        "test_psnr": [],
        "train_ssim": [],
        "test_ssim": []
    }
    lr = params["lr"]

    if params["save_img"]:
        save_orig_img(train_loader, folder_path, name_sub_folder="train_images")
        save_orig_img(test_loader, folder_path, name_sub_folder="test_images")

    for epoch in range(params["epochs"]):

        if params["learn_vec_lr"]:
            lr = get_lr(epoch, params["lr_vec"], params["cum_epochs"])
            optimizer = build_optimizer(network, params["optimizer"], lr, params["weight_decay"])

        start_epoch = time.time()
        train_loss_epoch, train_psnr_epoch, train_ssim_epoch = train_epoch(
            epoch,
            network,
            train_loader,
            optimizer,
            params["batch_size"],
            params["img_dim"],
            params["n_masks"],
            device,
            log_path,
            folder_path,
            wb_flag,
            save_img=params["save_img"]
        )
        print_training_messages(epoch, train_loss_epoch, lr, start_epoch, log_path)

        test_loss_epoch, test_psnr_epoch, test_ssim_epoch = test_net(
            epoch,
            network,
            test_loader,
            params["n_masks"],
            device,
            log_path,
            folder_path,
            params["batch_size"],
            params["img_dim"],
            wb_flag,
            save_img=params["save_img"]
        )

        numerical_outputs = update_numerical_outputs(
            numerical_outputs,
            train_loss_epoch,
            test_loss_epoch,
            train_psnr_epoch,
            test_psnr_epoch,
            train_ssim_epoch,
            test_ssim_epoch
        )

    save_all_run_numerical_outputs(numerical_outputs, folder_path, wb_flag)
    print_and_log_message("Run Finished Successfully", log_path)

    #image_results_subplot(folder_path, data_set="train_images", epochs_to_show=[0, 1, 2, 5, 10, params["epochs"]])
    #image_results_subplot(folder_path, data_set="test_images", epochs_to_show=[0, 1, 2, 5, 10, params["epochs"]])


def save_img_train_test(epoch, train_loader, test_loader, network, params, optimizer, device, folder_path, log_path, wb_flag):
    _ = train_epoch(epoch, network, train_loader, optimizer, params["batch_size"], params["z_dim"],
                    params["img_dim"], params["n_masks"], device, log_path, folder_path, wb_flag,
                    save_img=True)
    _ = test_loss_epoch = test_net(epoch, network, test_loader, device, log_path, folder_path, params["batch_size"],
                                   params["z_dim"], params["img_dim"], params["cr"], wb_flag, save_img=True)


def build_network(pic_width, n_masks, model_name, device):

    model_class = getattr(NN_architecture, model_name)
    network = model_class(pic_width, n_masks)

    # Before starting a new forward/backward pass
    torch.cuda.empty_cache()

    print("Built model successfully.")

    return network.to(device)


def build_optimizer(network, optimizer, learning_rate, weight_decay):

    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def update_numerical_outputs(numerical_outputs, train_loss_epoch, test_loss_epoch, train_psnr_epoch, test_psnr_epoch, train_ssim_epoch, test_ssim_epoch):
    numerical_outputs["train_loss"].append(train_loss_epoch)
    numerical_outputs["test_loss"].append(test_loss_epoch)
    numerical_outputs["train_psnr"].append(train_psnr_epoch)
    numerical_outputs["test_psnr"].append(test_psnr_epoch)
    numerical_outputs["train_ssim"].append(train_ssim_epoch)
    numerical_outputs["test_ssim"].append(test_ssim_epoch)
    return numerical_outputs


def get_lr(epoch, lr_vec, cum_epochs):
    for i, threshold in enumerate(cum_epochs):
        if epoch < threshold:
            return lr_vec[i]


def get_test_images(folder_path):
    orig_img_path = folder_path + "/test_images/orig_imgs_tensors.pt"
    all_images_tensor = torch.load(orig_img_path)
    return all_images_tensor


