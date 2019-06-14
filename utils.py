# -*- coding:utf-8-*-
import os
import math
import shutil       #shutil则就是对os中文件操作的补充。--移动 复制  打包 压缩 解压
import logging
import numpy as np

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

#https://www.cnblogs.com/CJOKER/p/8295272.html
class Logger(object):
    def __init__(self,log_file_name,log_level,logger_name):
        #第一步，创建一个logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        #第二步，创建一个handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        #第三步,定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s '
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        #第四步,将Hander添加到logger中
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    #numel()函数：返回数组中元素的个数

def data_augmentation(config,is_train=True):
    aug = []
    if is_train:
        #random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size,padding=4))
        #horizontal flip
        if config.augmentation.random_horizontal_flip:
            aug.append(transforms.RandomHorizontalFlip())
    aug.append(transforms.ToTensor())
    #normalize [-mean/std]
    if config.augmentation.normalize:
        if config.dataset =='cifar10':
            aug.append(transforms.Normalize(
                (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
            ))
        else:
            aug.append(transforms.Normalize(
                (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
            ))

    if is_train and config.augmentation.cutout:
        #cutout
        aug.append(Cutout(n_holes=config.augmentation.holes,
                          length=config.augmentation.length))
    return aug

class Cutout(object):
    def __init__(self,n_holes,length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h,w),np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            #np.clip()函数将将数组中的元素限制在a_min, a_max之间
            y1 = np.clip(y-self.length//2,0,h)   #" / "  表示浮点数除法，返回浮点结果," // " 表示整数除法,返回不大于结果的一个最大的整数
            y2 = np.clip(y+self.length//2,0,h)
            x1 = np.clip(x-self.length//2,0,w)
            x2 = np.clip(x+self.length//2,0,w)

            mask[y1:y2, x1:x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)          #把一个tensor变成和函数括号内一样形状的tensor，用法与expand（）类似
        img = img*mask

        return img

def save_checkpoint(state,is_best,filename):
    torch.save(state,filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename+'.pth.tar',filename+'_best.pth.tar')

def load_checkpoint(path,model,optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'],strict=False)

        if optimizer != None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=== load state dict done.also load optimizer from checkpoint'{}'(epoch{})".format(
                path,last_epoch+1
            ))

            return best_prec,last_epoch

def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == 'cifar10' or config.dataset == 'cifar100'
    if config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, transform=transform_train, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, transform=transform_test, download=True)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, transform=transform_train, download=True
        )
        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, transform=transform_test, download=True
        )

    train_loader = Data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    test_loader = Data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=True, num_workers=config.workers)
    return train_loader, test_loader

def mixup_data(x,y,alpha,device):
    '''Returns mixed inputs,pairs of targets,and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)       #返回一个0到n-1的数组

    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a,y_b = y,y[index]
    return mixed_x,y_a,y_b,lam

def mixup_criterion(criterion,pred,y_a,y_b,lam):
    return lam*criterion(pred,y_a)+(1-lam)*criterion(pred,y_b)

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer,epoch,config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
             (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)*\
             (1.0+math.cos(math.pi*ratio))/2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_schedule.min_lr + \
             (config.lr_scheduler.base_lr-config.lr_scheduler.min_lr)*\
             (1.0-math.tanh(config.lr_scheduler.lower_bound+
                            (config.lr_scheduler.upper_bound - config.lr_scheduler.lower_bound)*ratio))/2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
