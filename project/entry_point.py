import torch
import torch.nn as nn
import time

import apex

from model import ResNet50

def skyline_model_provider():
    return ResNet50().cuda()


def skyline_input_provider(batch_size=4):
    return (
        torch.randn((batch_size, 3, 128, 128)).cuda(),
        torch.randint(low=0, high=1000, size=(batch_size,)).cuda(),
    )

import torchvision
import torchvision.transforms as transforms

def get_loaders(train_bs, val_bs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((128, 128))
    ])

    # STEP #1:  Reorder Dataloader
    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.GaussianBlur(3),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     transforms.Resize((128, 128))
    # ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=val_bs, shuffle=False)
    
    return trainloader, testloader


def skyline_iteration_provider(model):
    # ---------------------------------------------
    # STEP 4: Fused optimizer
    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # ---------------------------------------------

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainloader, testloader = get_loaders(72, 72)

    def iteration(*args):
        # STEP 2: Manipulate batch size. Comment the below to load from input provider
        inputs, targets = args
        bs = inputs.shape[0]
        
        # ---------------------------------------------
        # Comment these:
        inputs, targets = next(iter(trainloader))
        inputs = inputs[:bs, :, :, :].to(torch.device("cuda"))
        targets = targets[:bs].to(torch.device("cuda"))
        # ---------------------------------------------

        optimizer.zero_grad()

        # ---------------------------------------------
        # STEP 3: Automatic Mixed Precision
        # with torch.autocast(device_type="cuda"):
        # ---------------------------------------------
        out = model(inputs)
        loss = loss_fn(out, targets)

        loss.backward()
        optimizer.step()

    return iteration

