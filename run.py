from ast import arg
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random
from loguru import logger
from torch.cuda.amp import autocast
from model_loader import load_model
import warnings
import TITAN
from torch.cuda.amp import autocast
warnings.filterwarnings("ignore")


def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 


def run():
    seed_torch()
    args = load_config()

    if args.tag == 'CUB200':
        args.rootpath = '/hdd/DataSet/CUB200'
        args.num_class = 100
        args.warm_up = 4
        from cub200_typical import load_data 
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.method, args.num_class, args.rootpath,args.batch_size,args.num_workers, args.noise_rate, args.noise_type)

    elif args.tag == 'CARS196':
        args.rootpath = '/hdd/DataSet/CARS196/cars_annos.mat'
        args.warm_up = 6
        args.num_class = 98
        from cars196_official import load_data
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.method, args.num_class, args.rootpath,args.batch_size,args.num_workers, args.noise_rate, args.noise_type)

    elif args.tag == 'FLICKR25K':
        args.rootpath = '/hdd/DataSet/Flickr25k/'
        args.num_class = 24
        from flickr25k import load_data
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.method, args.num_class, args.rootpath,args.batch_size,args.num_workers, args.noise_rate, args.noise_type)
    
    elif args.tag == 'CIFAR10':
        args.rootpath = '/hdd/DataSet/cifar10'
        args.num_class = 10
        args.warm_up = 8
        args.max_iter = 50
        from cifar10 import load_data
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.method, args.num_class, args.rootpath, args.batch_size ,args.num_workers, args.noise_rate, args.noise_type)
    
    # real-noise
    elif args.tag == 'Cars98N':
        args.rootpath = '/hdd/DataSet/CARS196/cars_annos.mat'
        args.num_class = 98
        args.warm_up = 6
        from cars98N import load_data
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.method, args.num_class, args.rootpath, args.batch_size,args.num_workers, args.noise_rate, args.noise_type)
    else:
        raise ValueError('No such Dataset')


    logger.add(os.path.join('final_logs', '{}_{}_{}_{}.log'.format(
        args.method, args.tag, args.noise_rate, args.noise_type
    )), rotation="500 MB", level="INFO")
    logger.info(args)

    if args.method == 'TITAN':
        TITAN.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args
        )
    elif args.method == 'PRISM':
        PRISM.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args
        )
    elif args.method == 'REL':
        REL.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args
        )
    
    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    parser = argparse.ArgumentParser(description='Noisy_Dense_Retrieval')
    # 【CUB200】
    # parser.add_argument('--tag', type=str, default='CUB200', help="Tag")

    #【CARs196】
    # parser.add_argument('--tag', type=str, default='CARS196', help="Tag")
  
    # 【Flickr25k】
    # parser.add_argument('--tag', type=str, default='FLICKR25K', help="Tag")

    # 【Cars98N】
    parser.add_argument('--tag', type=str, default='Cars98N', help="Tag")

    # 【cifar10】
    # parser.add_argument('--tag', type=str, default='CIFAR10', help="Tag")


    parser.add_argument('-n', '--noise_rate', default=0., type=float,
                        help='noisy rate.(default: 0.2); for cars98n, it equal 0.')

    parser.add_argument('-nt', '--noise_type', default='symmetric', type=str,
                        help='noisy type.(default: pairflip)')

    parser.add_argument('-m', '--method', default='PURE', type=str,
                        help='method to train the model.(default: PURE)')

    parser.add_argument('-k', '--topk', default=10, type=int,
                        help='Calculate map of top k.(default: 10)')
    
    parser.add_argument('-T', '--max-iter', default=30, type=int,
                        help='Number of iterations.(default: 30)')

    parser.add_argument('-l', '--lr', default=5e-5, type=float,
                        help='Learning rate.(default: 5e-5)')
    
    parser.add_argument('-wd', '--weight_decay', default=0.0005, type=float,
                        help='weight_decay.(default: 0.0005)')

    parser.add_argument('-w', '--num-workers', default=16, type=int,
                        help='Number of loading data threads.(default: 12)')

    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')

    parser.add_argument('-a', '--arch', default='resnet18', type=str,
                        help='CNN architecture.(default: resnet18)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')

    parser.add_argument('--train', action='store_true',
                        help='Training mode.')

    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluation mode.')

    parser.add_argument('-g', '--gpu', default=1, type=int,
                        help='Using gpu.')
    
    parser.add_argument('-e', '--evaluate-interval', default=2, type=int,
                        help='Interval of evaluation.(default: 2)') 

    parser.add_argument('-ls', '--last_stride', default=1, type=int,
                        help='For ResNet18') 
    
    parser.add_argument('-hd', '--hidden_dim', default=512, type=int,
                        help='Hidden_Dim feature for Image') 

    parser.add_argument('-ic', '--in_channels', default=1000, type=int,
                        help='2048/1000 for ResNet50/18') 
    
    parser.add_argument('-mbs', '--xbmsize', default=2000, type=int,
                        help='Size of MemoryBank') 

    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Hyper-parameter.(default:0.1)')
    
    parser.add_argument('--warm_up', default=2, type=float,
                        help='Hyper-parameter in pure.(2 for cub200 | cars98n; 6 for cars)')

    parser.add_argument('--window_size', default=5, type=float,
                        help='Hyper-parameter in prism')

    parser.add_argument('--num_gradual', default=4, type=float,
                        help='Hyper-parameter in prism')               

    args = parser.parse_args()
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    return args



if __name__ == '__main__':
    run()
