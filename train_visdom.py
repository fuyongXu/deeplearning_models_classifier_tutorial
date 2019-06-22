# -*-coding:utf-8-*-
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from visdom import Visdom

from easydict import EasyDict
from models import *
from utils import *

parser = argparse.ArgumentParser(description="Pytorch_Image_classifier_tutorial")
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path+'/log.txt',log_level=logging.DEBUG,logger_name='CIFAR').get_log()

#visdom = Visdom()



def train(train_loader,net,criterion,optimizer,epoch,device):
    global visdom

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info("===Epoch:[{}/{}]===".format(epoch+1, config.epochs))
    step = 0
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs,targets,config.mixup_alpha,device)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        step += 1
        #zero the gradient buffers
        optimizer.zero_grad()
        #backward()
        loss.backward()
        #update weight
        optimizer.step()

        #count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)            #这里_代表我们不关心的部分，而我们关系predicted部分。这部分对应了所属Label的索引https://cloud.tencent.com/developer/article/1433941
        total += targets.size(0)
        if config.mixup:
            correct += (lam*predicted.eq(targets_a).sum().item()+(1-lam)*predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        visdom.line([[train_loss, correct]], [step], win='train_loss', update='append')
        if(batch_index + 1) %100 == 0:
            logger.info("  === step:[{:3}/{}],train_loss:{:.3f}|train_acc:{:6.3f}%|lr:{:.6f}".format(
                batch_index+1, len(train_loader), train_loss/(batch_index+1), 100*correct/total, get_current_lr(optimizer)
            ))
    logger.info("  ===step:[{:3}/{}],train_loss:{:.3f}|train_acc:{:6.3f}%|lr:{:.6f}".format(
        batch_index+1, len(train_loader), train_loss/(batch_index+1), 100.0*correct/total, get_current_lr(optimizer)
    ))

    end  = time.time()
    logger.info("  ===cost time:{:.4f}s".format(end-start))
    train_loss = train_loss/(batch_index+1)
    train_acc = correct/total


    return train_loss, train_acc

#val
def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, visdom

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info("===== Validate =====".format(epoch+1,config.epochs))

    step = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            step += 1
    visdom.line([[test_loss, correct]], [step], win='test', update='append')
    visdom.images(inputs, win='x')
    visdom.text(str(predicted.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))

    logger.info("  ===test loss:{:.3f}|test acc{:6.3f}%".format(test_loss/(batch_index + 1), 100.0*correct/total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total


    #Save checkpoint
    acc = 100.*correct/total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec, visdom

    visdom = Visdom()
    visdom.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
    visdom.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc']))

    #read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)

    #convert to dict
    config = EasyDict(config)           #easydict的作用：可以使得以属性的方式去访问字典的值
    logger.info(config)

    #denfine net
    net = get_model(config)
    logger.info(net)
    logger.info("===total parameters:" + str(count_parameters(net)))

    #GPU or CPU
    device = 'cuda' if config.use_gpu else 'cpu'
    #data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.to(device)

    #define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),config.lr_scheduler.base_lr,
                                momentum=config.optimize.momentum,
                                weight_decay=config.optimize.weight_decay,
                                nesterov=config.optimize.nesterov)

    #resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(ckpt_file_name, net, optimizer)

    #load training data,do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config)
    )
    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False)
    )

    train_loader, test_loader = get_data_loader(transform_train,transform_test,config)

    #start training
    logger.info("        =======  start  training   ======     ")
    for epoch in range(last_epoch+1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    logger.info("=====Training Finished.   best_test_acc:{:.3f}%====".format(best_prec))

if __name__ == "__main__":
    main()