# Author : Pramil Paudel
import re
import matplotlib
from stego.StegoUtlis import compute_metrics
from stego.HidingNetwork import SwinHidingNet
import torch.nn.functional as F

matplotlib.use("Agg")
import pickle
import argparse
import os
import shutil
import socket
import time
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import gan.GanTrainerHR as G_trainer
import gan.GanTester as G_tester
import gan.GanUtils as G_utils
import lensless.lenslessConverter as lenslessConverter
import stego.Transform as transforms
from stego.RevealNetwork import RevealNetDualInput as RevealNet
from stego.CustomLoss import CombinedLoss, CenterPriorityMAELossTwoBoxes, CenterFocusedLossThree
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--outckpts', default='./training/', help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/', help='folder to output images')
parser.add_argument('--outcodes', default='./training/', help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.10, help='hyper parameter of beta')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')

######## Reading information about Data location and GPU availability ##################################################

print("CHECKING IF THE GPU IS AVAILABLE IN THE SYSTEM !!!")
gpu_available = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != "cpu":
    DATA_BASE_FOLDER = "/scratch/p522p287/DATA/STEN_DATA/DIV2K_STEN/"
    batch_size = 8
else:
    DATA_BASE_FOLDER = "C://Users/p522p287/Documents/PhDCS/data/"
    # DATA_BASE_FOLDER = "/Users/patthar/Documents/Projects/DATA/steg_data/"
    batch_size = 4

# TRAINED_MODEL_DIRECTORY_FOR_TESTING = "/scratch/p522p287/CODE/LiGANS/training/same_psf_result/"
TRAINED_MODEL_DIRECTORY_FOR_TESTING = "/scratch/p522p287/CODE/LiGANS/training/g020_Apr_28_2025_02-36-29_AM_/"
TRAINED_MODEL_DIRECTORY_FOR_GAN = "/scratch/p522p287/CODE/LiGANS/training/g020_Apr_29_2025_10-22-09_PM_/"

OUTPUT_GRAPH_PATH = DATA_BASE_FOLDER + 'output/'
MODELS_PATH = DATA_BASE_FOLDER + 'models/'
VALID_PATH = DATA_BASE_FOLDER + 'valid/'

TRAIN_COVER_PATH = DATA_BASE_FOLDER + 'train_cover/'
TRAIN_SECRET_PATH = DATA_BASE_FOLDER + 'train_secret/'

VALIDATION_COVER_PATH = DATA_BASE_FOLDER + 'validation_cover/'
VALIDATION_SECRET_PATH = DATA_BASE_FOLDER + 'validation_secret/'

TEST_COVER_PATH = DATA_BASE_FOLDER + 'test_cover/'
TEST_SECRET_PATH = DATA_BASE_FOLDER + 'test_secret/'

LOSS_PATH = DATA_BASE_FOLDER + 'loss/'

################################# SIZE OF THE IMAGE AND BATCH ##########################################################
processing_image_size = (128, 128)
secret_scene_processing_size = (64, 64)
# Tells the number of epoch to train hiding network
cut_off_epoch = 0
# Tells to load if you need epoch for the retrain
last_epoch = 0
# Tells if you want load previous Hiding and Revealing Models
retrain = True


def get_transforms(normalize=True):
    base_transforms = [
        transforms.Resize(size=processing_image_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(base_transforms)


def load_data():
    # Training loaders
    train_cover_loader = DataLoader(
        datasets.ImageFolder(TRAIN_COVER_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    train_secret_loader = DataLoader(
        datasets.ImageFolder(TRAIN_SECRET_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    # Validation loaders
    validation_cover_loader = DataLoader(
        datasets.ImageFolder(VALIDATION_COVER_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    validation_secret_loader = DataLoader(
        datasets.ImageFolder(VALIDATION_SECRET_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    # Test loaders
    test_cover_loader = DataLoader(
        datasets.ImageFolder(TEST_COVER_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    test_secret_loader = DataLoader(
        datasets.ImageFolder(TEST_SECRET_PATH, get_transforms(normalize=False)),
        batch_size=batch_size, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=True)

    return (train_cover_loader, train_secret_loader,
            validation_cover_loader, validation_secret_loader,
            test_cover_loader, test_secret_loader)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def custom_loss_function(output, target):
    weight = []
    with open('weight.pkl', 'rb') as f:
        weight = pickle.load(f)
    weight = torch.from_numpy(np.array(weight))
    weight = weight.to(dtype=torch.long, device=device)
    return (weight * (output - target) ** 2).sum() / weight.sum()


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)
    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main(train_cover_loader, train_secret_loader, validation_cover_loader, validation_secret_loader, test_cover_loader,
         test_secret_loader):
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss
    global image_save_frequency
    image_save_frequency = 1
    opt = parser.parse_args()
    ################# Safe Logging Name Setup #################
    # Generate safe timestamp
    cur_time = time.strftime('%b_%d_%Y_%I-%M-%S_%p', time.localtime())

    # Clean hostname and remark (in case they include unsafe characters)
    safe_hostname = re.sub(r'[<>:"/\\|?*, ]', '_', opt.hostname)
    safe_remark = re.sub(r'[<>:"/\\|?*, ]', '_', opt.remark)
    safe_time = re.sub(r'[<>:"/\\|?*, ]', '_', cur_time)

    # Create safe experiment dir
    experiment_dir = f"{safe_hostname}_{safe_time}_{safe_remark}"
    print("This is experimental direction !!!!!!", experiment_dir)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_dir))

    ################# Output configuration #################
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    cudnn.benchmark = True

    ############  Create dirs to save the result #############
    if not opt.test:
        try:
            opt.outckpts = os.path.join(opt.outckpts, experiment_dir, "checkPoints")
            opt.validationpics = os.path.join(opt.validationpics, experiment_dir, "validationPics")
            opt.outlogs = os.path.join(opt.outlogs, experiment_dir, "trainingLogs")
            opt.outPics = os.path.join(opt.outPics, experiment_dir, "output")

            os.makedirs(opt.outckpts, exist_ok=True)
            os.makedirs(opt.trainpics, exist_ok=True)
            os.makedirs(opt.outlogs, exist_ok=True)

        except OSError as e:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")
            print("Error:", e)
    ############ Create log file path and start logging #############
    logPath = os.path.join(opt.outlogs, f'{opt.dataset}_{opt.batchSize}_log.txt')
    print_log(str(opt), logPath)
    # save_current_codes(opt.outcodes)
    # Network Creation
    # Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    # Hnet = TransformerHidingNetwork(input_nc=6, output_nc=3, patch_size=4, embed_dim=128, num_layers=6, num_heads=8)
    Hnet = SwinHidingNet(in_channels=6, mid_channels=128, out_channels=3, img_size=128, patch_size=2, num_blocks=6)
    Hnet.to(device)
    Hnet.apply(weights_init)
    # whether to load pre-trained model
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).to(device)
    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.to(device)
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).to(device)
    print_network(Hnet)
    print_network(Rnet)
    # loading existing last epoch
    if retrain is True:
        print("LOADING DATA FROM THE CHECKPOINTS")
        opt.Hnet = TRAINED_MODEL_DIRECTORY_FOR_TESTING + "checkPoints/netH.pth"
        opt.Rnet = TRAINED_MODEL_DIRECTORY_FOR_TESTING + "checkPoints/netR.pth"
        Hnet.to(device)
        Rnet.to(device)
        Hnet.load_state_dict(torch.load(opt.Hnet, map_location='cpu'))
        Rnet.load_state_dict(torch.load(opt.Rnet, map_location='cpu'))
    model_G, model_D, opt_G, opt_D, scheduler_G, scheduler_D, losses, iter_losses, last_epoch = G_utils.load_models(
        netG='unet',
        netD='wgan',
        chkpoint=0,
        learning_rate=0.0001,
        device=device,
        test_mode=opt.test,
        len_data=len(train_cover_loader),
        batch_size=batch_size,
        ngpu=1,
        test_directory=TRAINED_MODEL_DIRECTORY_FOR_GAN + "checkPoints/")
    criterion = nn.MSELoss().to(device)
    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        adjusted_epoch = 0
        for epoch in range(last_epoch, opt.niter + last_epoch):
            epoch = epoch - adjusted_epoch
            ######################## train ##########################################
            model_G, model_D, G_loss, D_loss = train(train_cover_loader, train_secret_loader, model_G, model_D, opt_G,
                                                     opt_D,
                                                     scheduler_G, scheduler_D,
                                                     losses, iter_losses, epoch, Hnet=Hnet, Rnet=Rnet)
            ###################### validation  #####################################
            val_hloss, val_rloss, val_sumloss = validation(model_G, model_D, validation_cover_loader,
                                                           validation_secret_loader, epoch=epoch, Hnet=Hnet, Rnet=Rnet,
                                                           RESULT_DIR=TRAINED_MODEL_DIRECTORY_FOR_TESTING)

            ####################### adjust learning rate ############################
            schedulerH.step(val_sumloss)
            schedulerR.step(val_rloss)
            torch.save(Hnet.state_dict(), '%s/netH.pth' % (opt.outckpts))
            torch.save(Rnet.state_dict(), '%s/netR.pth' % (opt.outckpts))
        writer.close()
        scheduler_G.step()
        scheduler_D.step()  ### Testing even in train mode for temporarily
        test(test_cover_loader, test_secret_loader, model_G, model_D, Hnet, Rnet, criterion,
             TRAINED_MODEL_DIRECTORY_FOR_TESTING)
        print(
            "##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")
    # test mode
    else:
        print("LOADING DATA FROM THE CHECKPOINTS")
        opt.Hnet = TRAINED_MODEL_DIRECTORY_FOR_TESTING + "checkPoints/netH.pth"
        opt.Rnet = TRAINED_MODEL_DIRECTORY_FOR_TESTING + "checkPoints/netR.pth"
        Hnet.to(device)
        Rnet.to(device)
        Hnet.load_state_dict(torch.load(opt.Hnet, map_location='cpu'))
        Rnet.load_state_dict(torch.load(opt.Rnet, map_location='cpu'))
        test(test_cover_loader, test_secret_loader, model_G, model_D, Hnet, Rnet, criterion,
             TRAINED_MODEL_DIRECTORY_FOR_TESTING)
        print(
            "##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def resize_to_64(img):
    img = np.squeeze(img)

    # Handle both grayscale and color
    if img.ndim == 2:  # (H, W)
        img_resized = resize(img, (64, 64), anti_aliasing=True, preserve_range=True)
    elif img.ndim == 3:
        if img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # to (H, W, C)
        img_resized = resize(img, (64, 64), anti_aliasing=True, preserve_range=True)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    return img_resized


def plot_grid_triplet(cover_out, container_out, secret_out, reco_out, original_gt, num_samples=4, save_path=None,
                      metrics=None):
    titles = ["Cover", "Container", "Secret", "Recovered Secret", "Partial Reconstruction", "Difference (Cont - Cov)"]
    tensors = [cover_out, container_out, secret_out, reco_out, original_gt]

    # Compute the Difference
    diff_tensors = container_out - cover_out
    tensors.append(diff_tensors)

    fig, axes = plt.subplots(nrows=7, ncols=num_samples, figsize=(4 * num_samples, 17), dpi=100)
    plt.subplots_adjust(hspace=0.3, wspace=0.05)

    def safe_tensor_to_image(t):
        if t.ndim == 3 and t.shape[0] in [1, 3]:
            t = t.detach().cpu().numpy()
            t = np.transpose(t, (1, 2, 0))
        elif t.ndim == 3:
            t = t.detach().cpu().numpy()
        else:
            t = np.squeeze(t.detach().cpu().numpy())

        h, w = t.shape[:2]
        if min(h, w) < 10:
            print(f"Skipping small image of shape: {t.shape}")
            return None

        # Normalize safely
        t = np.clip((t - t.min()) / (t.max() - t.min() + 1e-8), 0, 1)
        return t

    # Plot images
    for row_idx, (img_set, row_axes) in enumerate(zip(tensors, axes[:-1])):
        for col_idx, ax in enumerate(row_axes):
            ax.axis("off")
            if col_idx < len(img_set):
                img_np = safe_tensor_to_image(img_set[col_idx])
                if img_np is not None:
                    ax.imshow(img_np)
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}", fontsize=12, pad=10)

    # Plot metrics neatly in last row
    if metrics:
        for col_idx, ax in enumerate(axes[-1]):
            if col_idx < len(metrics):
                psnr_val = float(metrics[col_idx].get("PSNR", 0.0))
                ssim_val = float(metrics[col_idx].get("SSIM", 0.0))
                mse = float(metrics[col_idx].get("MSE", 0.0))
                rmse = float(np.sqrt(mse))
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}\nMSE: {mse:.5f}\nRMSE: {rmse:.5f}",
                    fontsize=9, ha='center', va='center', wrap=True
                )

    # Add bold row labels
    for i, label in enumerate(titles):
        axes[i, 0].annotate(
            label, xy=(0, 0.5), xytext=(-axes[i, 0].yaxis.labelpad - 25, 0),
            xycoords='axes fraction', textcoords='offset points',
            ha='right', va='center', fontsize=11, fontweight='bold'
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grid saved at: {save_path}")
    plt.close(fig)


def train(train_cover_loader, train_secret_loader, model_G, model_D, opt_G, opt_D, scheduler_G, scheduler_D, losses,
          iter_losses, epoch, Hnet, Rnet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()
    NormRlosses = AverageMeter()  # Track normalized R loss

    stage = "Stage 2 - Gnet only" if epoch >= cut_off_epoch else "Stage 1 - Hnet & Rnet"
    print(f"Epoch {epoch}: {stage}")
    Hnet.train() if epoch < cut_off_epoch else Hnet.eval()
    Rnet.train() if epoch < cut_off_epoch else Rnet.eval()
    Rnet.use_center_weight = epoch < cut_off_epoch

    start_time = time.time()
    # contrib_logs = []

    for i, (cover_batch, secret_batch) in enumerate(zip(train_cover_loader, train_secret_loader)):
        cover_batch, secret_batch = (b[0] if isinstance(b, (list, tuple)) else b for b in [cover_batch, secret_batch])
        train_covers = cover_batch.to(device)
        train_secrets = secret_batch.to(device)

        if epoch < cut_off_epoch:
            Hnet.zero_grad()
            Rnet.zero_grad()

            train_secrets = F.interpolate(train_secrets, size=secret_scene_processing_size, mode='bilinear',
                                          align_corners=False)
            lensless_secret = lenslessConverter.convert_into_lensless(train_secrets).to(train_covers.device)
            concat_img = torch.cat((train_covers, lensless_secret), dim=1)
            container_img = Hnet(concat_img)

            loss_H = CombinedLoss(factors=(0.6, 0.2, 0.2)).to(device)
            errH = loss_H(container_img, train_covers)
            Hlosses.update(errH.item(), train_covers.size(0))

            container_img_detached = container_img.detach()
            rev_secret_img = Rnet(container_img_detached, lensless_secret)
            rev_secret_img_rec = lenslessConverter.partial_reconstruct_tensor_rev(rev_secret_img).to(device)

            # saving the working configuration
            loss_R = CenterFocusedLossThree(
                square_size=64,
                weight_inner=20.0,
                weight_background=0.01,
                center_mae_weight=0.6,
                global_l1_weight=0.05,
                perceptual_weight=1.5,
                debug=False
            ).to(device)

            # loss_R = CenterFocusedLossThree(
            #     square_size=64,
            #     weight_inner=20.0,
            #     weight_background=0.01,
            #     center_mae_weight=0.6,
            #     global_l1_weight=0.05,
            #     perceptual_weight=0,
            #     debug=False
            # ).to(device)

            errR = loss_R(
                y_pred=rev_secret_img,
                y_true=lensless_secret,
                perceptual_pred=rev_secret_img_rec,
                perceptual_target=train_secrets
            )
            Rlosses.update(errR.item(), train_covers.size(0))

            # Normalize and scale R loss
            normalized_errR = errR / (Rlosses.avg + 1e-6) * Hlosses.avg
            NormRlosses.update(normalized_errR.item(), train_covers.size(0))

            err_sum = errH + opt.beta * normalized_errR

            # Backward and optimizer steps
            err_sum.backward(retain_graph=True)
            optimizerH.step()
            errR.backward()
            optimizerR.step()

            SumLosses.update(errH.item() + errR.item(), train_covers.size(0))

            if epoch % 5 == 0 and i == 0:
                with torch.no_grad():
                    # Normalize for metric computation
                    norm_container = torch.clamp(container_img_detached, 0, 1)
                    norm_cover = torch.clamp(train_covers, 0, 1)

                    # Compute perceptual quality metrics
                    metrics = compute_metrics(norm_cover, norm_container)

                # Plot grid and show metrics under each image
                plot_grid_triplet(
                    cover_out=train_covers,
                    container_out=container_img_detached,
                    secret_out=lensless_secret,
                    reco_out=rev_secret_img,
                    original_gt=rev_secret_img_rec,
                    num_samples=5,
                    save_path=f"{opt.outPics}/train_epoch_{epoch}_{i}.png",
                    metrics=metrics
                )

        else:
            train_covers, train_secrets = train_secrets, train_covers
            train_secrets_lensless = F.interpolate(train_secrets, size=secret_scene_processing_size, mode='bilinear',
                                                   align_corners=False)
            lensless_secret = lenslessConverter.convert_into_lensless(train_secrets_lensless).to(device)
            # train_secrets = lenslessConverter.reconstruct_tensor_rev(lensless_secret).to(device)
            with torch.no_grad():
                concat_img = torch.cat((train_covers, lensless_secret), dim=1)
                container_img = Hnet(concat_img)
                rev_secret_img = Rnet(container_img.detach(), lensless_secret)

            rev_secret_img_rec = lenslessConverter.partial_reconstruct_tensor_rev(rev_secret_img).to(device)
            model_G, model_D, G_loss, D_loss = G_trainer.generator_network_train(
                model_G,
                model_D,
                opt_G,
                opt_D,
                scheduler_G,
                scheduler_D,
                train_covers.float(),
                container_img.float(),
                rev_secret_img.float(),
                rev_secret_img_rec.float(),
                train_secrets.float(),
                epoch,
                i,
                device,
                opt.outPics)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

    # --- Epoch Summary ---
    epoch_log = (
        f"[Epoch {epoch}/{opt.niter}] "
        f"H_loss: {Hlosses.avg:.4f} | "
        f"R_loss: {Rlosses.avg:.4f} | "
        f"Norm_R_loss: {NormRlosses.avg:.4f} | "
        f"Total: {SumLosses.avg:.4f}"
    )
    print_log(epoch_log, logPath)

    if not opt.debug:
        writer.add_scalar('train/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('train/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('train/norm_R_loss_avg', NormRlosses.avg, epoch)
        writer.add_scalar('train/sum_loss_avg', SumLosses.avg, epoch)
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
    return model_G, model_D, G_loss, D_loss


def validation(model_G, model_D, val_cover_loader, val_secret_loader, epoch, Hnet, Rnet, RESULT_DIR):
    print("\n############################## VALIDATION BEGIN ##############################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    # Backup and set center weighting for RevealNet
    if hasattr(Rnet, 'use_center_weight'):
        original_center_flag = Rnet.use_center_weight
        Rnet.use_center_weight = True
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()

    with torch.no_grad():
        for i, (cover_batch, secret_batch) in enumerate(zip(val_cover_loader, val_secret_loader)):
            cover_batch = cover_batch[0] if isinstance(cover_batch, (list, tuple)) else cover_batch
            secret_batch = secret_batch[0] if isinstance(secret_batch, (list, tuple)) else secret_batch

            test_covers = cover_batch.to(device)
            test_secrets = secret_batch.to(device)
            test_secrets_lensless = F.interpolate(test_secrets, size=secret_scene_processing_size, mode='bilinear',
                                                  align_corners=False)
            lensless_secret = lenslessConverter.convert_into_lensless(test_secrets_lensless).to(test_covers.device)
            # test_secrets = lenslessConverter.reconstruct_tensor_rev(lensless_secret).to(device)

            # ------------------ Hnet Forward ------------------ #
            concat_img = torch.cat((test_covers, lensless_secret), dim=1)
            container_img = Hnet(concat_img)

            # Hiding loss
            loss_H = CombinedLoss(factors=(0.6, 0.2, 0.2)).to(device)
            errH = loss_H(container_img, test_covers)
            Hlosses.update(errH.item(), test_covers.size(0))

            # ------------------ Rnet Forward ------------------ #
            container_img_detached = container_img.detach()
            rev_secret_img = Rnet(container_img_detached, lensless_secret)
            rev_secret_img_rec = lenslessConverter.partial_reconstruct_tensor_rev(rev_secret_img).to(test_covers.device)

            # RevealNet loss
            loss_R = CenterFocusedLossThree(
                square_size=64,
                weight_inner=20.0,
                weight_background=0.01,
                center_mae_weight=0.6,
                global_l1_weight=0.05,
                perceptual_weight=1.5,
                debug=False
            ).to(device)

            errR = loss_R(
                y_pred=rev_secret_img,
                y_true=lensless_secret,
                perceptual_pred=rev_secret_img_rec,
                perceptual_target=test_secrets
            )
            Rlosses.update(errR.item(), test_covers.size(0))

            # ------------------ GAN Evaluation ------------------ #
            if opt.test == "":
                result_dir = opt.outPics
            else:
                result_dir = RESULT_DIR
            G_tester.test_step(
                model_G=model_G,
                model_D=model_D,
                cover_images=test_covers.float(),
                container_images=container_img.float(),
                secret_input=rev_secret_img_rec.float(),
                gt_secret=test_secrets.float(),
                batch=i,
                batch_size=batch_size,
                RESULTS_DIR=result_dir,
                device=device,
                epoch=epoch,
                isValidation=True
            )

    # ------------------ Compute Final Losses ------------------ #
    val_hloss = Hlosses.avg
    raw_rloss = Rlosses.avg
    normalized_rloss = raw_rloss / (Rlosses.avg + 1e-6) * Hlosses.avg
    val_sumloss = val_hloss + opt.beta * normalized_rloss
    val_time = time.time() - start_time

    val_log = (
        f"Validation Epoch {epoch} Summary:\n"
        f"→ Avg_H_loss   = {val_hloss:.6f}\n"
        f"→ Avg_R_loss   = {raw_rloss:.6f}\n"
        f"→ Normalized_R = {normalized_rloss:.6f}\n"
        f"→ Total Loss   = {val_sumloss:.6f}\n"
        f"→ Duration     = {val_time:.2f} sec"
    )
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', val_hloss, epoch)
        writer.add_scalar('validation/R_loss_avg', raw_rloss, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    # Restore original RevealNet center weighting
    if hasattr(Rnet, 'use_center_weight'):
        Rnet.use_center_weight = original_center_flag
    print("############################## VALIDATION END ##############################\n")
    return val_hloss, raw_rloss, val_sumloss


def test(test_cover_loader, test_secret_loader, model_G, model_D, Hnet, epoch, Rnet, RESULT_DIR):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()

    Hnet.float()
    Rnet.float()
    Hnet.eval()
    Rnet.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, (cover_batch, secret_batch) in enumerate(zip(test_cover_loader, test_secret_loader)):
            # Handle dataloader formats
            if isinstance(cover_batch, (list, tuple)):
                cover_batch = cover_batch[0]
            if isinstance(secret_batch, (list, tuple)):
                secret_batch = secret_batch[0]

            test_covers = cover_batch.to(device)
            test_secrets = secret_batch.to(device)

            lensless_secret = lenslessConverter.convert_into_lensless(test_secrets).to(test_covers.device)

            # Hnet forward pass
            concat_img = torch.cat((test_covers, lensless_secret), dim=1)
            container_img = Hnet(concat_img)

            # Hnet loss
            combined_loss_fn_H = CombinedLoss(factors=(0.6, 0.2, 0.2))
            errH = combined_loss_fn_H(container_img, test_covers)
            Hlosses.update(errH.item(), test_covers.size(0))

            # Rnet forward pass
            container_img_detached = container_img.detach()
            rev_secret_img = Rnet(container_img_detached, lensless_secret)

            # Rnet loss (center + combined)
            center_priority_loss = CenterPriorityMAELossTwoBoxes()
            combined_loss_fn_R = CombinedLoss(factors=(0.5, 0.5, 0.0))
            errR1 = combined_loss_fn_R(rev_secret_img, test_secrets)
            errR2 = center_priority_loss(rev_secret_img, test_secrets)
            errR = errR1 + errR2
            Rlosses.update(errR.item(), test_covers.size(0))

            rev_secret_img_rec = lenslessConverter.partial_reconstruct_tensor_rev(rev_secret_img).to(test_covers.device)
            if opt.test == "":
                result_dir = opt.outPics
            else:
                result_dir = RESULT_DIR
            G_tester.test_step(
                model_G=model_G,
                model_D=model_D,
                cover_images=test_covers.float(),
                container_images=container_img.float(),
                secret_input=rev_secret_img_rec.float(),
                gt_secret=test_secrets.float(),
                batch=i,
                batch_size=batch_size,
                RESULTS_DIR=result_dir,
                device=device,
                epoch=epoch,
                isValidation=True
            )

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss
    val_time = time.time() - start_time

    val_log = "test[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t test time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss


def print_log(log_info, log_path, console=True):
    # Print to console
    if console:
        print(log_info)

    # Skip file writing if in debug mode
    if opt.debug:
        return

    # Write to log file in UTF-8 encoding
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+', encoding='utf-8') as f:
        f.write(log_info + '\n')


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    train_cover_loader, train_secret_loader, validation_cover_loader, validation_secret_loader, test_cover_loader, test_secret_loader = load_data()
    main(train_cover_loader, train_secret_loader, validation_cover_loader, validation_secret_loader, test_cover_loader,
         test_secret_loader)
    print("completed !!")
