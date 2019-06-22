# pytorch_image_classifier_tutorial
This repository uses Pytorch to implement the popular CNN architectures, using the dataset CIFAR. The following is the reference paper:

## Architecture
 * (lenet)[LeNet-5, convolutional neural networks](http://yann.lecun.com/exdb/lenet/)
 * (alexnet) [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
 * (vgg) [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
 * (resnet) [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
 * (preresnet) [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1512.03385)
  * (resnext) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
 * (densenet) [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
 * (senet) [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
 * (bam) [BAM: Bottleneck Attention Module](https://arxiv.org/abs/1807.06514)
 * (cbam) [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
 * (genet) [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://arxiv.org/abs/1810.12348)
 * (sknet) [SKNet: Selective Kernel Networks](https://arxiv.org/abs/1903.06586)
 * (mobilenetV1)[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861?context=cs)
 * (mobilenetV2)[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
 
 ## Regularization
 * (shake-shake) [Shake-Shake regularization](https://arxiv.org/abs/1705.07485)
 * (cutout) [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
 * (mixup) [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

## Learning Rate Scheduler
* (cos_lr) [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
* (htd_lr) [Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification](https://arxiv.org/abs/1806.01593)

## Requirements
*  Python >= 3.5
* PyTorch = 0.4 or 1.0
* Tensorboard (if you want to use the tensorboard for visualization)
* pyyaml, easydict, tensorboardX

## Usage
  Run the command for training as following:
  >##1 GPU for lenet  
  python -u train.py --work-path ./experiments/cifar10/lenet  
  ##resume from checkpoint  
  python -u train.py --work-path ./experiments/cifar10/lenet --resume  
  ##2 GPUs for resnet1202  
  CUDA_VISIBLE_DEVICES=0,1 python -u train.py --work-path ./experiments/cifar10/preresnet20  
  ##4 GPUs for densenet190bc  
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --work-path ./experiments/cifar10/densenet100bc  
  
Feel free to contact me if you have any suggestions or questions, issues are welcome,
create a PR if you find any bugs or you want to contribute.:smile:


