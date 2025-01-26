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
ap.add_argument("--goal", type=str, choices=["predMulti", "sepMulti", "peakWave"], default="predMulti")
ap.add_argument("--resdownDictFile", type=str, default=None)
ap.add_argument("--adam", action="store_true")
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

#from datsimx.ml_sep.unet import unet_model
from datsimx.ml_sep.dset_loader import WaveData, MlatData, NlattsData
from datsimx.ml_sep.arch import predictMulti
from datsimx.ml_sep import arch

logname = os.path.join(args.outdir, "logfile.txt")
logger = get_logger(logname)

def dice_loss(pred, lab, reduction="mean"):
    """ checks if `pred` lattice is any of the `lab` lattices.. we only want model to separate one lattice at a time"""

    pred_rep = pred.repeat((1,lab.shape[1],1,1))
    numer = (pred_rep*lab).sum(axis=-1).sum(axis=-1)
    denom= pred_rep.sum(axis=-1).sum(axis=-1) + lab.sum(axis=-1).sum(axis=-1)
    dice_metric = 1-(2*numer)/(denom)
    dloss = torch.min(dice_metric, axis=-1).values
    if torch.sum(dloss < 0) > 0:
        from IPython import embed;embed()
    if reduction == "mean":
        dloss = dloss.mean()
    return dloss


class diceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, lab):
        numer = (pred[:,0]*lab[:,0]).sum(axis=-1).sum(axis=-1)
        denom= pred[:,0].sum(axis=-1).sum(axis=-1) + lab[:,0].sum(axis=-1).sum(axis=-1)
        dloss = 1-2*numer/denom
        dloss = dloss.mean()
        return dloss

def one_to_one_dice_loss(pred, lab, reduction="mean"):
    numer = (pred[:,0]*lab[:,0]).sum(axis=-1).sum(axis=-1)
    denom= pred[:,0].sum(axis=-1).sum(axis=-1) + lab[:,0].sum(axis=-1).sum(axis=-1)
    dloss = 1-2*numer/denom
    if reduction == "mean":
        dloss = dloss.mean()
    return dloss

    

from itertools import permutations
class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.ind_orders3 = set(permutations([1,2,3]))
        self.ind_orders2 = set(permutations([1,2]))
        self.ind_orders1 = set(permutations([1]))
        
    def forward(self, pred, lab):
        nlat = lab.max().item()
        #assert nlat in {1,2,3}
        ind_orders = getattr(self, f"ind_orders{nlat}")
        one_hot = torch.nn.functional.one_hot(lab)
        losses = []
        for ijk in ind_orders:
            ijk_lab = torch.argmax(one_hot[:,:,:,:,[0]+list(ijk)], dim=-1)
            ijk_loss = self.loss(pred, ijk_lab.float())
            losses.append(ijk_loss)
        return torch.tensor(losses).min()

dev = "cuda:0"
if args.cpu:
    dev = "cpu"

sig = torch.nn.Sigmoid()
if args.goal=="predMulti":
    model = predictMulti()
    model = torch.nn.Sequential(model, sig)

    train_data = NlattsData(args.trainf, n_ex=2000)

    test_data = NlattsData(args.testf, n_ex=100)
    val_data = NlattsData(args.trainf, n_ex=100)
    loss = torch.nn.BCELoss()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m)

elif args.goal == "sepMulti":

    # using a pre-trained resnetDown portion of the model?
    if args.resdownDictFile is not None:
        logger.info("Loading partial model.. ")
        resdown = arch.resnetDown()
        trained_dict = torch.load(args.resdownDictFile, weights_only=True)
        resdown_dict = resdown.state_dict()
        trained_dict = {k: v for k, v in trained_dict.items() if k in resdown_dict}
        resdown_dict.update(trained_dict)
        load_stdout = resdown.load_state_dict(resdown_dict)

        # fix the resdown params...
        fix_names = []
        for name,p in resdown.named_parameters():
            fix_names.append(name)
            p.requires_grad = False
        fix_names = set(fix_names)
        
        # load the U-net
        model = arch.resnetU(down=resdown)
        model = torch.nn.Sequential(model, sig)

        # get the new list of params we are saving ... 
        train_params = []
        for name, p in model.named_parameters():
            if name in fix_names:
                p.requires_grad = False
                continue
            else:
                train_params.append(p)
    else:
        model = arch.resnetU()
        model = torch.nn.Sequential(model, sig)
        train_params = model.parameters()

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m)

    mlc=1#0.75  # multi lattice chance, else shot is single lattice ... 
    train_data = MlatData(args.trainf, n_ex=2000, mlat_chance=mlc)
    test_data = MlatData(args.testf, n_ex=100, mlat_chance=mlc)
    val_data = MlatData(args.trainf, n_ex=100, mlat_chance=mlc)
    loss =dice_loss 

else:
    #model = arch.resnetU()
    #model = torch.nn.Sequential(model, sig)
    model = arch.FCN50()

    train_data = WaveData(args.trainf)
    test_data = WaveData(args.testf,250) 
    val_data = WaveData(args.trainf, maximg=len(test_data))

    loss = diceLoss() #one_to_one_dice_loss 
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m)

train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.nwork)
test_loader = DataLoader(test_data, batch_size=1)
val_loader = DataLoader(val_data, batch_size=1)
ntrain_batch = len(train_loader)
model = model.to(dev)

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

