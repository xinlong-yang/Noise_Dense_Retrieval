from re import A
from tracemalloc import is_tracing
from PIL import Image, ImageFile
import torch
import numpy as np
import os
import sys
import pickle
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import scipy.io as scio
from PIL import ImageFilter
from loguru import logger
from noise import noisify_dataset
import pandas as pd


def get_img():
    find_img=pd.read_csv('/hdd/YXL_Project/Noise_Dense_Retrieval/prepare_CARS_98N/CARS_98N_list.csv').to_numpy()
    find_img=[['https://i.pinimg.com/736x/'+j[0][len('https://i.pinimg.com/736x/'):],j[1]] for j in find_img]
    return find_img


class_num = 98
class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Onehot(object):
    def __call__(self, sample, num_classes=98):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1
        return target_onehot

def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def train_aug_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

def query_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_data(method, class_num, rootpath, batch_size, num_workers, noisy_rate, noise_type, task = 'retrieval'):
    cars98N.init(rootpath, task)
    query_dataset = cars98N(method, class_num, noise_type, 'query', query_transform(), target_transform=Onehot(), aug_transform=train_aug_transform(), noisy_rate=noisy_rate)
    train_dataset = cars98N(method, class_num, noise_type, 'train', train_transform(), target_transform=Onehot(), aug_transform=train_aug_transform(), noisy_rate=noisy_rate)
    retrieval_dataset = cars98N(method, class_num, noise_type, 'retrieval', query_transform(), target_transform=Onehot(),aug_transform=train_aug_transform(),noisy_rate=noisy_rate)
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
        drop_last=False
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class cars98N(Dataset):
    def __init__(self, method, class_num, noise_type, mode, transform=None, target_transform=None, aug_transform=None, noisy_rate = None, noise = True):
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.aug_transform = aug_transform
        self.method = method

        if mode == 'train':
            self.data = cars98N.TRAIN_DATA
            self.targets = cars98N.TRAIN_TARGETS

        elif mode == 'query':
            self.data = cars98N.QUERY_DATA
            self.targets = cars98N.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = cars98N.RETRIEVAL_DATA
            self.targets = cars98N.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            img_t = self.transform(img)
            
        return img_t, self.target_transform(self.targets[index]), index

    def __len__(self):
        return self.data.shape[0]
    
    def get_targets_onehot(self):
        one_hot = torch.zeros((self.targets.shape[0],class_num))
        for i in range(self.targets.shape[0]):
            one_hot[i,:] = self.target_transform(self.targets[i])
        return  one_hot

    def get_targets(self):
        return self.targets

    @staticmethod
    def init(rootpath, task):
       
        train_data = []
        train_label = []
        test_data = []
        test_label = []
       
        image_path = {}
        annos = scio.loadmat(rootpath)
        images = annos['annotations'][0]
        num_images = images.size
        

        image_real = pd.read_csv('/hdd/YXL_Project/Noise_Dense_Retrieval/prepare_CARS_98N/CARS_98N_list.csv').to_numpy()
        for j in image_real:
            path_img = '/hdd/DataSet/cars_noise/img_v3/' + str(j[1]) + '/' + j[0].split('/')[-1]
            if path_img.endswith('gif'):
                continue
            train_data.append(path_img)
            train_label.append(int(j[1]))
            # print(path_img,j[1])


        for i in range(num_images):
            class_id = images[i][5][0][0]
            image_path = '/hdd/DataSet/CARS196/' + images[i][0][0]
            if class_id <= 98:
                # train_data.append(image_path)
                # train_label.append(int(class_id) - 1)
                continue
            else:
                test_data.append(image_path)
                test_label.append(int(class_id) - 99)

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        test_data = np.array(test_data)
        test_label = np.array(test_label)

        if task == 'retrieval':
            cars98N.QUERY_DATA = test_data
            cars98N.QUERY_TARGETS = test_label

            cars98N.RETRIEVAL_DATA = test_data
            cars98N.RETRIEVAL_TARGETS = test_label

            cars98N.TRAIN_DATA = train_data
            cars98N.TRAIN_TARGETS = train_label

            logger.info('Query Num: {}'.format(cars98N.QUERY_DATA.shape[0]))
            logger.info('Retrieval Num: {}'.format(cars98N.RETRIEVAL_DATA.shape[0]))
            logger.info('Train Num: {}'.format(cars98N.TRAIN_DATA.shape[0]))

