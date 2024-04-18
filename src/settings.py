import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Orientation Prediction Trainging")

    parser.add_argument("-d", "--data-dir", default="../data_pkl/", help="dataset directory")
    parser.add_argument("-p", "--phase", default="train", help="train or test")
    parser.add_argument("-w", "--workers", default=8, type=int, help="number of data loading workers")
    
    parser.add_argument("-c", "--ckp-name", default="debug", type=str, help="name of checkpoint")
    parser.add_argument("-m", "--ckp-message", default="", type=str, help="message of checkpoint")
    
    parser.add_argument("--gpu-id", default='0', type=str, help="gpu device(s) used")
    
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    
    parser.add_argument("--weight-decay", default=1e-6, type=float, help="weight decay")
    
    parser.add_argument("--optim", default='sgd', type=str, help="optimizer used")
    
    parser.add_argument("--loss-t", default="MSE", type=str, help="losses used")
    parser.add_argument("--loss-a", default="MSE", type=str, help="losses used")
    parser.add_argument("--loss-d", default="MaskedMSE", type=str, help="losses used")
    parser.add_argument("--loss-defog", default="MaskedMSE", type=str, help="losses used")
    parser.add_argument("--loss-vis", default="MaskedMSE", type=str, help="losses used")
    
    parser.add_argument("--lambda-t", default=1, type=float, help="lambda used")
    parser.add_argument("--lambda-a", default=1, type=float, help="lambda used")
    parser.add_argument("--lambda-d", default=1, type=float, help="lambda used")
    parser.add_argument("--lambda-defog", default=1, type=float, help="lambda used")
    parser.add_argument("--lambda-vis", default=1, type=float, help="lambda used")
    
    parser.add_argument("--model-name", default="DMRVisNet", type=str, help="model name")
    
    parser.add_argument("--train-batch", default=2, type=int, help="batch size of train phase")
    parser.add_argument("--test-batch", default=2, type=int, help="batch size of test phase")
    
    parser.add_argument("--height", default=576//2, type=int, help="image height")
    parser.add_argument("--width", default=1024//2, type=int, help="image width")
    parser.add_argument("--t_thresh", default=-1, type=float, help="pixels that is lower than this threshold will be gradient-free")
    parser.add_argument("--eps", default=1e-8, type=float, help="a small number to avoid x/0 or log(0)")
    
    parser.add_argument("--resume", action="store_true", default=False, help="resume model or not")
    parser.add_argument("--resume-name", default="", type=str, help="resume model")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch id")
    parser.add_argument("--max-epochs", default=200, type=int, help="number of total epochs")
    
    parser.add_argument("--seed", default=2021, type=int, help="mannual seed")
    parser.add_argument("--no-cuda", dest='use_cuda', action="store_false", default=True)
    parser.add_argument("--ckp-dir", default="../checkpoints/", type=str, help="directory of checkpoint")
    
    args = parser.parse_args()
        
    args.resume_name = os.path.join(args.ckp_dir, args.resume_name)
        
    if args.phase == "train":
        if args.ckp_name == "debug":
            args.workers = 0
            args.batchsize = 1
        
        args.ckp = os.path.join(args.ckp_dir, args.ckp_name)
        if not os.path.exists(args.ckp):
            os.makedirs(args.ckp)

        with open(os.path.join(args.ckp, "config.txt"), 'a') as fh:
            fh.write(args.__str__())

    return args
