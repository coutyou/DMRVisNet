import os
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as nnf

__all__ = [
    "FoggyDataset",
]

class FoggyDataset(Dataset):
    def __init__(self, data_path="/media/user1/YJ/data_pkl", phase='train', train_rate=0.7, valid_rate=0.9, height=576//2, width=1024//2):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.phase = phase
        self.height = height
        self.width = width
        
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.min_ratio = 0.75
        self.max_ratio = 1
        self.init_height = 576
        self.init_width = 1024
        
        self.pkls = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.pkls.sort()
       
        if phase == 'train':
            self.pkls = self.pkls[0:int(len(self.pkls) * train_rate)]
        elif phase == 'valid':
            self.pkls = self.pkls[int(len(self.pkls) * train_rate):int(len(self.pkls) * valid_rate)]
        else:
            self.pkls = self.pkls[int(len(self.pkls) * valid_rate):]
            
        print("DataSet Init done! images num: {}".format(len(self.pkls)))

    def __getitem__(self, idx):
        pkl_path = self.pkls[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if self.phase == "train":
            trans_3ch = {"normalize": False, 
                         "horizontalflip": True, 
                         "totensor": True, 
                         "tofloat": True, 
                         "resize": self.height!=self.init_height or self.width!=self.init_width,
                         "crop": True,
                        }
            trans_1ch = {"normalize": False, 
                         "horizontalflip": True, 
                         "totensor": True, 
                         "tofloat": True, 
                         "resize": self.height!=self.init_height or self.width!=self.init_width,
                         "crop": True,
                        }
            p_flip = np.random.random()
            p_crop = [np.random.random(), np.random.random(), np.random.random()]
        else:
            trans_3ch = {"normalize": False, 
                         "horizontalflip": False, 
                         "totensor": True, 
                         "tofloat": True, 
                         "resize": self.height!=self.init_height or self.width!=self.init_width,
                         "crop": False,
                        }
            trans_1ch = {"normalize": False, 
                         "horizontalflip": False, 
                         "totensor": True, 
                         "tofloat": True, 
                         "resize": self.height!=self.init_height or self.width!=self.init_width,
                         "crop": False,
                        }
            p_flip = None
            p_crop = None
        
        if trans_1ch["resize"] == True:
            data["SkyMask"] = data["SkyMask"].astype(np.uint8)
        
        if self.phase == "train":
            data["Scene"] = Image.fromarray((data["Scene"] * 255).astype(np.uint8))
            data["Scene"] = transforms.ColorJitter(brightness=self.brightness, 
                                                   contrast=self.contrast, 
                                                   saturation=self.saturation, 
                                                   hue=self.hue
                                                  )(data["Scene"])
            data["Scene"] = np.array(data["Scene"]).astype(np.float64) / 255
            
        for key in ["Scene", "FoggyScene_0.05"]:
            data[key] = self.data_augmentation(data[key], trans_3ch, p_flip, p_crop)
        for key in ["DepthPerspective", "t_0.05", "SkyMask"]:
            data[key] = self.data_augmentation(data[key], trans_1ch, p_flip, p_crop)
            
        if trans_1ch["resize"] == True:    
            data["SkyMask"] = data["SkyMask"].bool()    

        data["DepthPerspective"] = 1 / data["DepthPerspective"]
        data["DepthPerspective"][data["SkyMask"]] = 0
        data["t_0.05"][data["SkyMask"]] = 0
        
        data["Visibility"] = torch.FloatTensor([data["Visibility"]])
        data["A"] = torch.from_numpy(data["A"]).type(torch.FloatTensor)
        
        if self.phase == "train":
            data["FoggyScene_0.05"] = data["Scene"]*data["t_0.05"] + data["A"][:,None,None].repeat(1, self.height, self.width)*(1 - data["t_0.05"])
        
        return data

    def __len__(self):
        return len(self.pkls)
    
    def data_augmentation(self, img, trans, p_flip, p_crop):
        # RandomHorizontalFlip
        if trans["horizontalflip"] and p_flip > 0.5:
            img = np.flip(img, 1).copy()
        # ToTensor
        if trans["totensor"]:
            img = transforms.Compose([transforms.ToTensor()])(img)
        # RandomCrop
        if trans["crop"]:
            scale = self.min_ratio + p_crop[0] * (self.max_ratio - self.min_ratio)
            new_h = int(self.init_height * scale)
            new_w = int(self.init_width * scale)
            start_h = int(p_crop[1] * (self.init_height - new_h))
            start_w = int(p_crop[2] * (self.init_width - new_w))
            img = img[:, start_h:start_h+new_h, start_w:start_w+new_w]
        # Resize
        if trans["resize"]:
            img = nnf.interpolate(img[None,:], size=(self.height, self.width), mode='nearest')[0]
        # ToFloat
        if trans["tofloat"]:
            if img.dtype != torch.bool:
                img = img.type(torch.FloatTensor)
        return img
    