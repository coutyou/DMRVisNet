import os
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from settings import parse_args
from utils import AverageMeter, Logger, savefig, Bar
from dataset import FoggyDataset
from models import generate_model, save_ckp, load_ckp
from losses import generate_loss, generate_optim, generate_scheduler

def train(data_loader, model, optim, criterions, args):
    model.train()
    
    loss_am_a = AverageMeter()
    loss_am_t = AverageMeter()
    loss_am_d = AverageMeter()
    loss_am_defog = AverageMeter()
    loss_am_vis = AverageMeter()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    bar = Bar(f'Epoch {epoch}:  training', max=len(data_loader))
    end = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
            
        batch_size = batch["FoggyScene_0.05"].shape[0]
        
        if args.use_cuda:
            batch["FoggyScene_0.05"] = batch["FoggyScene_0.05"].cuda(non_blocking=True)
            batch["t_0.05"] = batch["t_0.05"].cuda(non_blocking=True)
            batch["A"] = batch["A"].cuda(non_blocking=True)
            batch["DepthPerspective"] = batch["DepthPerspective"].cuda(non_blocking=True)
            batch["Scene"] = batch["Scene"].cuda(non_blocking=True)
            batch["Visibility"] = batch["Visibility"].cuda(non_blocking=True)
            batch["SkyMask"] = batch["SkyMask"].cuda(non_blocking=True)
            
        batch["Visibility"] = repeat(math.log(0.05) / batch["Visibility"], 'b 1 -> b 1 h w', h=args.height, w=args.width)
        batch["A"] = rearrange(batch["A"], 'b c -> b c 1 1')
        
        optim.zero_grad()
        
        # output
        a, t, d, defog, vis = model(batch["FoggyScene_0.05"])
        
        # loss
        mask = (batch["t_0.05"] < args.t_thresh) | batch["SkyMask"]
        loss_a = criterions[0](a, batch["A"])
        if args.loss_t[:6] == "Masked":
            loss_t = criterions[1](t, batch["t_0.05"], mask)
        else:
            loss_t = criterions[1](t, batch["t_0.05"])
        if args.loss_d[:6] == "Masked":
            loss_d = criterions[2](d, batch["DepthPerspective"], mask)
        else:
            loss_d = criterions[2](d, batch["DepthPerspective"]) + 0.001 * criterions[5](d, batch["Scene"])
        if args.loss_defog[:6] == "Masked":
            loss_defog = criterions[3](defog, batch["Scene"], mask)
        else:
            loss_defog = criterions[3](defog, batch["Scene"])
        if args.loss_vis[:6] == "Masked":
            loss_vis = criterions[4](vis, batch["Visibility"], mask)
        else:
            loss_vis = criterions[4](vis, batch["Visibility"])
        
        # update
        loss_am_a.update(loss_a.item())
        loss_am_t.update(loss_t.item())
        loss_am_d.update(loss_d.item())
        loss_am_defog.update(loss_defog.item())
        loss_am_vis.update(loss_vis.item())
        
        # compute gradient and do SGD step
        (args.lambda_a*loss_a + args.lambda_t*loss_t + args.lambda_d*loss_d + args.lambda_defog*loss_defog + args.lambda_vis*loss_vis).backward()
        
        optim.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix = f"({batch_idx+1}/{len(data_loader)}) | ETA: {bar.eta_td:} | Loss_a: {loss_am_a.avg:.4f} | Loss_t: {loss_am_t.avg:.4f} | Loss_d: {loss_am_d.avg:.4f} | Loss_defog: {loss_am_defog.avg:.4f} | Loss_vis: {loss_am_vis.avg:.4f}"
        bar.next()
        
    bar.finish()
    
    return (loss_am_a.avg, loss_am_t.avg, loss_am_d.avg, loss_am_defog.avg, loss_am_vis.avg)

def val(data_loader, model, criterions, args):
    model.eval()
        
    loss_am_a = AverageMeter()
    loss_am_t = AverageMeter()
    loss_am_d = AverageMeter()
    loss_am_defog = AverageMeter()
    loss_am_vis = AverageMeter()
 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    bar = Bar(f"Epoch {epoch}:  validing", max=len(data_loader))
    end = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        batch_size = batch["FoggyScene_0.05"].shape[0]
        if args.use_cuda:
            batch["FoggyScene_0.05"] = batch["FoggyScene_0.05"].cuda(non_blocking=True)
            batch["t_0.05"] = batch["t_0.05"].cuda(non_blocking=True)
            batch["A"] = batch["A"].cuda(non_blocking=True)
            batch["DepthPerspective"] = batch["DepthPerspective"].cuda(non_blocking=True)
            batch["Scene"] = batch["Scene"].cuda(non_blocking=True)
            batch["Visibility"] = batch["Visibility"].cuda(non_blocking=True)
            batch["SkyMask"] = batch["SkyMask"].cuda(non_blocking=True)
            
        batch["Visibility"] = repeat(math.log(0.05) / batch["Visibility"], 'b 1 -> b 1 h w', h=args.height, w=args.width)
        batch["A"] = rearrange(batch["A"], 'b c -> b c 1 1')
        
        # output
        a, t, d, defog, vis = model(batch["FoggyScene_0.05"])
    
        # loss
        mask = (batch["t_0.05"] < args.t_thresh) | batch["SkyMask"]
        loss_a = criterions[0](a, batch["A"])
        if args.loss_t[:6] == "Masked":
            loss_t = criterions[1](t, batch["t_0.05"], mask)
        else:
            loss_t = criterions[1](t, batch["t_0.05"])
        if args.loss_d[:6] == "Masked":
            loss_d = criterions[2](d, batch["DepthPerspective"], mask)
        else:
            loss_d = criterions[2](d, batch["DepthPerspective"]) + 0.001 * criterions[5](d, batch["Scene"])
        if args.loss_defog[:6] == "Masked":
            loss_defog = criterions[3](defog, batch["Scene"], mask)
        else:
            loss_defog = criterions[3](defog, batch["Scene"])
        if args.loss_vis[:6] == "Masked":
            loss_vis = criterions[4](vis, batch["Visibility"], mask)
        else:
            loss_vis = criterions[4](vis, batch["Visibility"])
        
        # update
        loss_am_a.update(loss_a.item())
        loss_am_t.update(loss_t.item())
        loss_am_d.update(loss_d.item())
        loss_am_defog.update(loss_defog.item())
        loss_am_vis.update(loss_vis.item())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix = f"({batch_idx+1}/{len(data_loader)}) | ETA: {bar.eta_td:} | Loss_a: {loss_am_a.avg:.4f} | Loss_t: {loss_am_t.avg:.4f} | Loss_d: {loss_am_d.avg:.4f} | Loss_defog: {loss_am_defog.avg:.4f} | Loss_vis: {loss_am_vis.avg:.4f}"
        bar.next()
        
    bar.finish()
    
    return (loss_am_a.avg, loss_am_t.avg, loss_am_d.avg, loss_am_defog.avg, loss_am_vis.avg)

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # model
    model = generate_model(args)
    print('=> Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters())/1e6))
    print("-"*100)
    
    # cuda
    if args.use_cuda:
        model.cuda()
    cudnn.benchmark = True

    # resume
    if args.resume:
        checkpoint = load_ckp(args.resume_name)
        if not checkpoint:
            raise ValueError('Failed to load checkpoint')
        else:
            print("-"*100)
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        # clear memory
        del checkpoint, model_dict

    # data_loader
    dataset_train = FoggyDataset(phase="train", data_path=args.data_dir, height=args.height, width=args.width)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True
    )

    dataset_valid = FoggyDataset(phase="valid", data_path=args.data_dir, height=args.height, width=args.width)
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.train_batch, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    print("-"*100)
    
    # logger
    logger = Logger(os.path.join(args.ckp, 'log.txt'), title=args.model_name)
    logger.set_names(['Learning Rate', 'Train Loss a', 'Train Loss t', 'Train Loss d', 'Train Loss defog', 'Train Loss vis', 'Valid Loss a', 'Valid Loss t', 'Valid Loss d', 'Valid Loss defog', 'Valid Loss vis'])
    
    # optimizer
    optim = generate_optim(args.optim, args.lr, args.weight_decay, model)
    
    # schedule
    scheduler = generate_scheduler(optim, patience=10, verbose=False)
    
    # criterion
    criterion_a = generate_loss(args.loss_a)
    criterion_t = generate_loss(args.loss_t)
    criterion_d = generate_loss(args.loss_d)
    criterion_defog = generate_loss(args.loss_defog)
    criterion_vis = generate_loss(args.loss_vis)
    criterion_smooth = generate_loss("Smooth")
    
    # train and val
    for epoch in range(args.start_epoch, args.max_epochs):
        begin_time = time.time()
        
        train_loss_a, train_loss_t, train_loss_d, train_loss_defog, train_loss_vis = train(data_loader_train, model, optim, [criterion_a, criterion_t, criterion_d, criterion_defog, criterion_vis, criterion_smooth], args)
            
        with torch.no_grad():
            val_loss_a, val_loss_t, val_loss_d, val_loss_defog, val_loss_vis = val(data_loader_valid, model, [criterion_a, criterion_t, criterion_d, criterion_defog, criterion_vis, criterion_smooth], args)

        scheduler.step(args.lambda_a*val_loss_a + args.lambda_t*val_loss_t + args.lambda_d*val_loss_d + args.lambda_defog*val_loss_defog + args.lambda_vis*val_loss_vis)
        
        logger.append([optim.param_groups[0]['lr'], train_loss_a, train_loss_t, train_loss_d, train_loss_defog, train_loss_vis, val_loss_a, val_loss_t, val_loss_d, val_loss_defog, val_loss_vis])

        save_ckp(model, args.model_name, optim, epoch+1, args)

        print(f"Epoch:{epoch}  time:{time.time() - begin_time:.3f}s\ntrain_loss_a={train_loss_a:.3f}  train_loss_t={train_loss_t:.3f}  train_loss_d={train_loss_d:.3f}  train_loss_defog={train_loss_defog:.3f}  train_loss_vis={train_loss_vis:.3f}\nvalid_loss_a={val_loss_a:.3f}  valid_loss_t={val_loss_t:.3f}  valid_loss_d={val_loss_d:.3f}  valid_loss_defog={val_loss_defog:.3f}  valid_loss_vis={val_loss_vis:.3f}")
        print("-"*100)
        
        if optim.param_groups[0]['lr'] <= (args.lr / 1e3):
            break

    # plot
    logger.close()
    logger.plot(logger.names[1::5])
    savefig(os.path.join(args.ckp, 'loss_log_a.jpg'))
    logger.plot(logger.names[2::5])
    savefig(os.path.join(args.ckp, 'loss_log_t.jpg'))
    logger.plot(logger.names[3::5])
    savefig(os.path.join(args.ckp, 'loss_log_d.jpg'))
    logger.plot(logger.names[4::5])
    savefig(os.path.join(args.ckp, 'loss_log_defog.jpg'))
    logger.plot(logger.names[5::5])
    savefig(os.path.join(args.ckp, 'loss_log_vis.jpg'))
    
    print("Finish!")