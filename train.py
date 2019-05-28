# -*-coding:utf-8-*-
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from easydict import EasyDict
from models import *
from utils import *

parser = argparse.ArgumentParser(description="Pytorch_Image_classifier_tutorial")
parser.add_argument('--work-path',required=True,type=str)
parser.add_argument('--resume',action='store_true',help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path+'/log.txt',log_level=logging.DEBUG,logger_name='CIFAR').get_log()

def train(train_loader,net,criterion,optimizer,epoch,device):
    global writer       #创建一个SummaryWriter实例

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info("===Epoch:[{}/{}]===".format(epoch+1,config.epochs))
    for batch_index,(inputs,targets) in enumerate(train_loader):
        inputs,targets = inputs.to(device),targets.to(device)
        if config.mixup:
            inputs,targets_a,targets_b,lam = mixup_data(inputs,targets,config.mixup_alpha,device)
            outputs = net(inputs)
            loss = mixup_criterion(criterion,outputs,targets_a,targets_b,lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs,targets)

        #zero the gradient buffers
        optimizer.zero_grad()
        #backward()
        loss.backword()
        #update weight
        optimizer.step()

        #count the loss and acc
        train_loss += loss.item()
        _,predicted = outputs.max(1)            #这里_代表我们不关心的部分，而我们关系predicted部分。这部分对应了所属Label的索引https://cloud.tencent.com/developer/article/1433941
        total += targets.size(0)
        if config.mixup:
            correct += (lam*predicted.eq(targets_a).sum().item()+(1-lam)*predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if(batch_index + 1) %100 == 0:
            logger.info("  === step:[{:3}/{}],train_loss:{:.3f}|train_acc:{:6.3f}%|lr:{:.6f}".format(
                batch_index+1,len(train_loader),train_loss/(batch_index+1),100*correct/total,get_current_lr(optimizer)
            ))
    logger.info("  ===step:[{:3}/{}],train_loss:{:.3f}|train_acc:{:6.3f}%|lr:{:.6f}".format(
        batch_index+1,len(train_loader),train_loss/(batch_index+1),100.0*correct/total,get_current_lr(optimizer)
    ))

    end  = time.time()
    logger.info("  ===cost time:{:.4f}s".format(end-start))
    train_loss = train_loss/(batch_index+1)
    train_acc = correct/total

    writer.add_scalar('train_loss',train_loss,epoch)
    writer.add_scalar('train_acc',train_acc,epoch)

    return train_loss,train_acc
