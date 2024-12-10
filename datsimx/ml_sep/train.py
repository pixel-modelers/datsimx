from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("trainf", type=str, help="train data input file")
ap.add_argument("testf", type=str, help="test data input file")
ap.add_argument("outdir", type=str)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--nep", type=int, default=100)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--nwork", type=int, default=1)
ap.add_argument("--savefreq", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--m", type=float, default=0)
args = ap.parse_args()

import logging


def get_logger(filename=None, level="info", do_nothing=False):
    """
    :param filename: optionally log to a file
    :param level: logging level of the console (info, debug or critical)
        INFO: Confirmation that things are working as expected.
        DEBUG: Detailed information, typically of interest only when diagnosing problems
        CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    :param do_nothing: return a logger that doesnt actually log (for non-root processes)
    :return:
    """
    levels = {"info": 20, "debug": 10, "critical": 50}
    if do_nothing:
        logger = logging.getLogger()
        logger.setLevel(levels["critical"])
        return logger
    logger = logging.getLogger("ml_sep.train")
    logger.setLevel(levels["info"])

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(levels[level])
    logger.addHandler(console)

    if filename is not None:
        logfile = logging.FileHandler(filename)
        logfile.setFormatter(logging.Formatter("%(asctime)s >>  %(message)s"))
        logfile.setLevel(levels["info"])
        logger.addHandler(logfile)
    return logger

import os
os.makedirs(args.outdir, exist_ok=True)
import numpy as np
import torch
from torch.utils.data import DataLoader

from datsimx.ml_sep.unet import unet_model
from datsimx.ml_sep.dset_loader import MlatData, NlattsData

from pytorch3d.loss import chamfer_distance

def dice_loss(pred, lab, reduction="mean"):
    lab_sum = lab.sum(axis=-1).sum(axis=-1).sum(axis=-1)
    numer1 = (pred*lab).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    denom1 = pred.sum(axis=-1).sum(axis=-2).sum(axis=-1) + lab_sum

    # switch order of lattices
    pred2 = pred[:,[1,0]]
    numer2 = (pred2*lab).sum(axis=-1).sum(axis=-1).sum(axis=-1)
    denom2 = pred2.sum(axis=-1).sum(axis=-2).sum(axis=-1) + lab_sum

    # take best-case loss
    dloss = 1-2*torch.max(numer1/denom1, numer2/denom2)
    if reduction == "mean":
        dloss = dloss.mean()
    return dloss


dev = "cuda:0"
if args.cpu:
    dev = "cpu"

from datsimx.ml_sep.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

from math import ceil
dim=832
#ws = ceil(7*(dim)/224)
#ws = 26
#model = SwinTransformerSys(img_size=dim, 
#                           num_classes=2, 
#                           window_size=ws, 
#                           in_chans=1)
#from resonet.params import res50
#model = res50()

from datsimx.ml_sep.arch import predictMulti
model = predictMulti()

#model = unet_model.UNet(1,2)
sig = torch.nn.Sigmoid()
model = torch.nn.Sequential(model, sig)

ml_chance=1
train_data = NlattsData(args.trainf, n_ex=2000)
train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nwork)

test_data = NlattsData(args.testf, n_ex=100)
test_loader = DataLoader(test_data, batch_size=2)

val_data = NlattsData(args.trainf, n_ex=100)
val_loader = DataLoader(val_data, batch_size=2)

loss = torch.nn.BCELoss()
#loss = dice_loss

#loss = torch.nn.L1Loss()
ntrain_batch = len(train_loader)
model = model.to(dev)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m)

logname = os.path.join(args.outdir, "logfile.txt")
logger = get_logger(logname)

logger.info(f"\nModel: {model}")
logger.info(f"\nOptimizer: {optimizer}")
logger.info(f"\nLoss: {loss}")
logger.info(f"\nargs: {args}")

for i_ep in range(args.nep):
    model.train()
    train_loss = []
    for i_batch, (dat, lab) in enumerate(train_loader):
        dat = dat.to(dev)
        lab = lab.to(dev)
        optimizer.zero_grad()
        pred = model(dat)
        L = loss(pred, lab)
        L.backward()
        optimizer.step()
        train_loss.append(L.item())
        logger.info(f"Train Loss={L.item():.6f} ep {i_ep+1} batch {i_batch+1}/{len(train_loader)} ")

    with torch.no_grad():
        model.eval()
        val_loss, test_loss = [],[]
        for losses, name, loader in  zip( [val_loss, test_loss], ["Val", "Test"], [val_loader, test_loader]):
            for i_batch, (dat, lab) in enumerate(loader):
                dat = dat.to(dev)
                lab = lab.to(dev)
                pred = model(dat)
                L = loss(pred, lab)
                losses.append(L.item())
                logger.info(f"{name} Loss={L.item():.6f} ep {i_ep+1} batch {i_batch+1}/{len(test_loader)} ")
        ave_test_l = np.mean(test_loss)
        ave_val_l = np.mean(val_loss)
        logger.info(f"End of Ep {i_ep+1} ; ave test loss = {ave_test_l:.6f} ; ave trainval loss = {ave_val_l:.6f}")
        
        if (i_ep+1) % args.savefreq == 0:
            save_name=os.path.join(args.outdir, f"model_Ep{i_ep+1}.net")
            torch.save(model.state_dict(), save_name)

