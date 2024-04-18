import os
import math

import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from settings import parse_args
from dataset import FoggyDataset
from models import generate_model, load_ckp

def get_class(data):
    if data < 200:
        res = 0
    elif data < 400:
        res = 1
    elif data < 600:
        res = 2
    elif data < 800:
        res = 3
    else:
        res = 4
    return res

def cal_metrics(pred, true, metric="AbsRel"):
    if metric == "AbsRel":
        sum_ = 0
        for i in range(len(pred)):
            sum_ += abs(pred[i] - true[i]) / true[i]
        return sum_ / len(pred)
    elif metric == "SqRel":
        sum_ = 0
        for i in range(len(pred)):
            sum_ += ((pred[i] - true[i]) / true[i]) ** 2
        return sum_ / len(pred)
    elif metric == "RMSE":
        sum_ = 0
        for i in range(len(pred)):
            sum_ += (pred[i] - true[i]) ** 2
        return math.sqrt(sum_ / len(pred))
    elif metric == "RMSElog":
        sum_ = 0
        for i in range(len(pred)):
            sum_ += (math.log(pred[i], 10) - math.log(true[i], 10)) ** 2
        return math.sqrt(sum_ / len(pred))
    elif metric == "Acc":
        sum_ = 0
        for i in range(len(pred)):
            if get_class(pred[i]) == get_class(true[i]):
                sum_ += 1
        return sum_ / len(pred)

def get_pred(data_loader, model, min_t=1e-2, min_dist=10, max_dist=1e5, eps=1e-8, sort=False):
    pred = []
    true = []

    for batch in tqdm(data_loader):
        batch_size = batch["FoggyScene_0.05"].shape[0]

        with torch.no_grad():
            _, t, _, _, vis = model(batch["FoggyScene_0.05"].cuda())
            
        for i in range(batch_size):
            vis_ = math.log(0.05) / (vis[i] + 1e-8)
            valid_vis = vis_[(t[i] > min_t) & (vis_.abs() < max_dist)]
            if len(valid_vis) == 0:
                pred.append(min_dist)
            else:
                pred.append(float(valid_vis.mean()))
            
            true.append(float(batch["Visibility"][i]))

    if sort:
        pred = [x for _, x in sorted(zip(true, pred), key=lambda pair: pair[0])]
        true.sort()
    
    return pred, true

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
    dataset_test = FoggyDataset(phase="test", data_path=args.data_dir, height=args.height, width=args.width)
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.test_batch, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    print("-"*100)
    
    model = model.eval()
    
    with torch.no_grad():
        pred_test, true_test = get_pred(data_loader_test, model, eps=args.eps)
    
    for metric in ["AbsRel", "SqRel", "RMSE", "RMSElog", "Acc"]:
        print(f"{metric} of test set: {cal_metrics(pred_test, true_test, metric):.5f}")
        print("-"*100)
    