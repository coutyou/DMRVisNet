import os
import torch

from .DMRVisNet import DMRVisNet

def generate_model(args):
    print(f'=> create model "{args.model_name}"')
    if args.model_name == 'DMRVisNet':
        model = DMRVisNet(eps=args.eps)
    else:
        raise ValueError(f'unsupport model architecture {args.model_name}')
    return model

def save_ckp(model, model_name, optimizer, epoch, args, save_ckp_num=3):
    model_name = f'epoch_{epoch}_{model_name}.pth.tar'
    ckp_record = os.path.join(args.ckp, f'checkpoint')
    if not os.path.isfile(ckp_record):
        with open(ckp_record, 'w') as fp:
            info_str = []
            info_str.append(f'model_name: {model_name}\n')
            fp.writelines(info_str)
    else:
        with open(ckp_record, 'r+') as fp:
            ckp_lines = [line.strip('\n') for line in fp.readlines()]
            if len(ckp_lines) >= save_ckp_num:
                for ckp_file in ckp_lines[save_ckp_num-1:]:
                    os.remove(os.path.join(args.ckp, ckp_file.split(' ')[-1]))
                ckp_lines = ckp_lines[:save_ckp_num-1]
            ckp_lines = [line + '\n' for line in ckp_lines]
            ckp_lines.insert(0, f'model_name: {model_name}\n')
            fp.seek(0, 0)
            fp.writelines(ckp_lines)
        
    model_save_path = os.path.join(args.ckp, model_name)
    torch.save(
        {
            'args': args,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_save_path
    )

def load_ckp(ckp):
    ckp_record = os.path.join(ckp, f'checkpoint')
    if os.path.isfile(ckp_record):
        with open(ckp_record, 'r') as fp:
            ckp_lines = [line.strip('\n') for line in fp.readlines()]
            model_name = ckp_lines[0].split(' ')[-1]
        try:
            ckp = torch.load(os.path.join(ckp, model_name))
        except FileNotFoundError:
            print(f'=> Can\'t found checkpoint "{model_name}"')
            return False
        print(f'=> load checkpoint "{model_name}" SUCCESS!')
        return ckp
    else:
        return False