import torch
from torch import nn
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import calc_cumu_ssim_batch, save_randomize_outputs, calc_cumu_psnr_batch


def test_net(epoch, network, loader, n_masks, device, log_path, folder_path, batch_size, img_dim,
             wb_flag, save_img=False):
    network.eval()
    network.to(device)
    cumu_loss, cumu_psnr, cumu_ssim = 0, 0, 0
    n_batchs = len(loader.batch_sampler)
    n_samples = n_batchs * batch_size
    pic_width = int(math.sqrt(img_dim))
    criterion = nn.MSELoss()

    for batch_index, sim_bucket_tensor in enumerate(loader):
        mes_vectors = sim_bucket_tensor[0].view(-1, 1, n_masks).to(device)
        orig_images = sim_bucket_tensor[1].view(-1, 1, img_dim).to(device)

        reconstruct_imgs_batch = network(mes_vectors)
        reconstruct_imgs_batch = reconstruct_imgs_batch.view(-1, 1, img_dim)
        loss = criterion(reconstruct_imgs_batch, orig_images)
        cumu_loss += loss.item()
        torch.cuda.empty_cache()  # Before starting a new forward/backward pass
        batch_psnr = calc_cumu_psnr_batch(reconstruct_imgs_batch, orig_images, pic_width)
        batch_ssim = calc_cumu_ssim_batch(reconstruct_imgs_batch, orig_images, pic_width)
        cumu_psnr += batch_psnr
        cumu_ssim += batch_ssim
        print_and_log_message(f"Epoch number {epoch}, batch number {batch_index}/{n_batchs}:"
                              f"       batch loss {loss.item()}", log_path)
        if wb_flag:
            wandb.log({"test_loss_batch": loss.item(), "test_psnr_batch": batch_psnr / batch_size,
                       "test_ssim_batch": batch_ssim / batch_size, "test_batch_index": batch_index})
        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, orig_images, int(math.sqrt(img_dim)),
                                   folder_path, 'test_images', wb_flag)

    test_loss, test_psnr, test_ssim = cumu_loss / n_samples, cumu_psnr / n_samples, cumu_ssim / n_samples
    if wb_flag:
        wandb.log({"epoch": epoch, 'test_loss': test_loss})

    return test_loss, test_psnr, test_ssim
