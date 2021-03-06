
# coding: utf-8

# In[2]:


from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np


# In[3]:


use_cuda = torch.cuda.is_available()


# In[4]:


use_cuda


# In[5]:



manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy


# In[6]:



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inp):
        outputs=[]
        x=inp
        module_list =list(self.features.modules())[1:]
        for l in module_list:
            
            x = l(x)
            outputs.append(x)
        
        y = x.view(inp.size(0), -1)
        o = self.classifier(y)
        return o,outputs[-1].view(inp.size(0), -1),outputs[-4].view(inp.size(0), -1)


def alexnet(classes):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(classes)
    return model


# In[7]:


class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes,num_layers=1):
        self.num_layers=num_layers
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.preds=nn.Sequential(
           nn.Linear(num_classes,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds2=nn.Sequential(
           nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.preds3=nn.Sequential(
           nn.Linear(1024,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            )
        self.correct=nn.Sequential(
           nn.Linear(1,num_classes),
            nn.ReLU(),
            nn.Linear(num_classes,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*(2+num_layers),256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,l,c,o,l1,l2):
        
        out_l = self.labels(l)
        out_c = self.correct(c)
        out_o = self.preds(o)
        out_l1 = self.preds2(l1)
        out_l2 = self.preds3(l2)
        
        _outs= torch.cat((out_c,out_l),1)
        
        if self.num_layers>0:
            _outs= torch.cat((_outs,out_o),1)
        if self.num_layers>1:
            _outs= torch.cat((_outs,out_l1),1)
            
        if self.num_layers>2:
            _outs= torch.cat((_outs,out_l2),1)
    
        is_member =self.combine(_outs )
        
        
        return self.output(is_member)


# In[8]:


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)[0]
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100==0:
            
            print ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))

    return (losses.avg, top1.avg)


# In[9]:


dataset='cifar100'
checkpoint_path='./checkpoints_100cifar_alexnet_white'
train_batch=100
test_batch=100
lr=0.05
epochs=500
state={}
state['lr']=lr


# In[53]:


def attack(trainloader,testloader, model,inference_model,classifier_criterion, criterion,classifier_optimizer ,optimizer, epoch, use_cuda,num_batchs=1000,is_train=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    inference_model.eval()
    if is_train:
        inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx,((tr_input, tr_target) ,(te_input, te_target)) in enumerate(zip(trainloader,testloader)):
        # measure data loading time
        if batch_idx > num_batchs:
            break
        data_time.update(time.time() - end)
        tr_input = tr_input.cuda()
        te_input = te_input.cuda()
        tr_target = tr_target.cuda()
        te_target = te_target.cuda()
        
        
        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input))
        
        pred_outputs,l1,l2 = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        mtop1, mtop5 =accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        
        mtop1_a.update(mtop1[0], model_input.size(0))
        mtop5_a.update(mtop5[0], model_input.size(0))

        
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0),100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        
        
        
        
        
        
        c= criterion_classifier(pred_outputs,infer_input).view([-1,1])#torch.autograd.Variable(torch.from_numpy(c.view([-1,1]).data.cpu().numpy()).cuda())
        
        
        
        preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())
        member_output = inference_model(infer_input_one_hot,c,pred_outputs,l1,l2)
        
        
        

        
        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data[0], model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if is_train:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx%10==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx ,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))

    return (losses.avg, top1.avg)


# In[54]:


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [20,40]:
        state['lr'] *= 0.1 
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


# In[55]:


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


# In[56]:


global best_acc
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(checkpoint_path):
    mkdir_p(checkpoint_path)



# Data
print('==> Preparing dataset %s' % dataset)
transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),


    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),


    ])
    
    


if dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100



# Model
print("==> creating model ")





# In[57]:


net= alexnet(num_classes)

net = torch.nn.DataParallel(net).cuda()


# In[58]:


resume='/mnt/nfs/work1/amir/milad/pretrained_models/alexnet_best'

checkpoint = torch.load(resume)
best_acc = checkpoint['best_acc']
start_epoch = checkpoint['epoch']
net.load_state_dict(checkpoint['state_dict'])



# In[59]:


net=list(net.children())[0]
net=net.cuda()


# In[60]:





128#inferenece_model = torch.nn.DataParallel(inferenece_model).cuda()

cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# Resume
title = 'cifar-100'


# In[61]:


criterion_attack = nn.MSELoss()


# In[62]:


criterion_classifier = nn.CrossEntropyLoss(reduce=False)


# In[63]:


trainset = dataloader(root='./data100', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=1)

testset = dataloader(root='./data100', train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=True, num_workers=1)


# In[48]:


epoch=300
test_loss, test_acc = test(testloader, net, criterion, epoch, use_cuda)
print (test_acc)


# In[64]:


batch_privacy=100
trainset = dataloader(root='./data100', train=True, download=True, transform=transform_train)
testset = dataloader(root='./data100', train=False, download=False, transform=transform_test)

r = np.arange(50000)
np.random.shuffle(r)

private_trainset_intrain = []
private_trainset_intest = []

private_testset_intrain =[] 
private_testset_intest =[] 


for i in range(25000):
    private_trainset_intrain.append(trainset[r[i]])


    

for i in range(25000,50000):
    private_testset_intrain.append(trainset[r[i]])

    
r = np.arange(10000)
np.random.shuffle(r)
  
for i in range(5000):
    private_trainset_intest.append(testset[r[i]])


for i in range(5000,10000):
    private_testset_intest.append(testset[r[i]])







private_trainloader_intrain = data.DataLoader(private_trainset_intrain, batch_size=batch_privacy, shuffle=True, num_workers=1)
private_trainloader_intest = data.DataLoader(private_trainset_intest, batch_size=batch_privacy, shuffle=True, num_workers=1)


private_testloader_intrain = data.DataLoader(private_testset_intrain, batch_size=batch_privacy, shuffle=True, num_workers=1)
private_testloader_intest = data.DataLoader(private_testset_intest, batch_size=batch_privacy, shuffle=True, num_workers=1)






# In[67]:


inferenece_model = InferenceAttack_HZ(100,num_layers=2)
inferenece_model = inferenece_model.cuda()
optimizer_mem = optim.Adam(inferenece_model.parameters(), lr=0.0001)


# In[68]:


best_acc=0
for epoch in range(100):
    attack(private_trainloader_intrain,private_trainloader_intest,net,inferenece_model,criterion_classifier,criterion_attack,optimizer,optimizer_mem,epoch,use_cuda,is_train=True)
    bb=attack(private_testloader_intrain,private_testloader_intest,net,inferenece_model,criterion_classifier,criterion_attack,optimizer,optimizer_mem,epoch,use_cuda,is_train=False)
    best_acc=max(bb[1],best_acc)
    print (best_acc)
    

