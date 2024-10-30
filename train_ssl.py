# -*- coding: utf-8 -*-
import os
import datetime
from optim.pretrain import *
import argparse
import torch
from utils.utils import get_config_from_json
from optim.train import supervised_train


'''
'MiddlePhalanxOutlineAgeGroup', 
'ProximalPhalanxOutlineAgeGroup', 
'SwedishLeaf', 
'MixedShapesRegularTrain', 
'Crop'
'''

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')
    parser.add_argument('--alpha', type=float, default=0.3, help='Past future split point')
    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='training patience')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')
    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    parser.add_argument('--stride', type=float, default=0.2,
                        help='stride for forecast model')
    parser.add_argument('--horizon', type=float, default=0.1,
                        help='horizon for forecast model')
    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    # optimization   
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='ECG200',
                        choices=['CricketX',
                                 'CricketY',
                                 'CricketZ',
                                 'ECG200',
                                 'ECG5000',
                                 'CBF',
                                 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound',
                                 'MFPT','XJTU',
                                 'EpilepticSeizure',
                                 'SwedishLeaf',
                                 'WordSynonyms',
                                 'ACSF1'
                                 ],
                        help='dataset')
    parser.add_argument('--nb_class', type=int, default=3,
                        help='class number')
    parser.add_argument('--n_class', type=int, default=2)
    # Hyper-parameters for vat model
    parser.add_argument('--n_power', type=int, default=4, metavar='N',
                        help='the iteration number of power iteration method in VAT')
    parser.add_argument('--xi', type=float, default=3, metavar='W', help='xi for VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='W', help='epsilon for VAT')
    parser.add_argument('--ucr_path', type=str, default='./datasets',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    parser.add_argument('--weight_rampup', default=30, type=int, metavar='EPOCHS',
                        help='the length of rampup weight (default: 30)')
    parser.add_argument('--usp_weight', default=1.0, type=float, metavar='W',
                        help='the upper of unsuperivsed weight (default: 1.0)')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4', help='Backbone model')
    parser.add_argument('--model_name', type=str, default='SSTSC', choices=['SupCE', 'SSTSC', 'MTL', 'Pi', 'train_SemiInterPF'], help='Model name')
    parser.add_argument('--config_dir', type=str, default='./config/', help='Config directory')
    parser.add_argument('--label_ratio', type=float, nargs='+', default=[0.1, 0.2, 0.4, 1.0], help='Label ratio')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    exp = 'exp-cls'
    Seeds = [0, 1, 2, 3, 4]
    Runs = range(10)

    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']

    # Load configuration for dataset
    config_dict = get_config_from_json(f'{opt.config_dir}/{opt.dataset_name}_config.json')
    opt.class_type = config_dict['class_type']
    opt.piece_size = config_dict['piece_size']

    # Set augmentation types
    if aug1 == aug2:
        opt.aug_type = aug1
    elif isinstance(aug1, list):
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    for label_ratio in opt.label_ratio:
        opt.label_ratio = label_ratio
        model_paras = f'label{opt.label_ratio}_{opt.alpha}' if opt.model_name == 'SemiPF' else f'label{opt.label_ratio}'

        log_dir = f'./results/{exp}/{opt.dataset_name}/{opt.model_name}/{model_paras}'

        os.makedirs(log_dir, exist_ok=True)

        file2print_detail_train = open(f"{log_dir}/train_detail.log", 'a+')
        print(datetime.datetime.now(), file=file2print_detail_train)
        print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_label\tAcc_unlabel\tEpoch_max", file=file2print_detail_train)

        file2print = open(f"{log_dir}/test.log", 'a+')
        print(datetime.datetime.now(), file=file2print)
        print("Dataset\tAcc_mean\tAcc_std\tEpoch_max", file=file2print)

        file2print_detail = open(f"{log_dir}/test_detail.log", 'a+')
        print(datetime.datetime.now(), file=file2print_detail)
        print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max", file=file2print_detail)

        ACCs = {}
        MAX_EPOCHs_seed = {}
        ACCs_seed = {}

        for seed in Seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            opt.ckpt_dir = f'./ckpt/{exp}/{opt.model_name}/{opt.dataset_name}/{"_".join(opt.aug_type)}/{model_paras}/{seed}'
            os.makedirs(opt.ckpt_dir, exist_ok=True)

            print(f'[INFO] Running at: {opt.dataset_name}')

            x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ = load_ucr2018(opt.ucr_path, opt.dataset_name)
            seed = 42
            acc_test = 0.85  # Example value, replace with actual logic
            acc_unlabel = 0.75  # Example value, replace with actual logic
            epoch_max = 100 
            ACCs_run = {}
            MAX_EPOCHs_run = {}
            for run in Runs:
                if opt.model_name == 'SupCE':
                    acc_test, epoch_max = supervised_train(x_train, y_train, x_val, y_val, x_test, y_test, opt)
                    acc_unlabel = 0
                elif 'MTL' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_MTL(x_train, y_train, x_val, y_val, x_test, y_test, opt)
                elif 'SSTSC' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_sstsc(x_train, y_train, x_val, y_val, x_test, y_test, opt)
                elif 'Pseudo' in opt.model_name:
                    acc_test, acc_unlabel, acc_ws, epoch_max = train_pseudo(x_train, y_train, x_val, y_val, x_test, y_test, opt)
                elif 'Pi' in opt.model_name:
                    acc_test, acc_unlabel, acc_ws, epoch_max = train_pi(x_train, y_train, x_val, y_val, x_test, y_test, opt)

                print(f"{opt.dataset_name}\t{x_train.shape[0]}\t{x_test.shape[0]}\t{x_train.shape[1]}\t{opt.nb_class}"
                      f"\t{seed}\t{round(acc_test, 2)}\t{round(acc_unlabel, 2)}\t{epoch_max}", file=file2print_detail_train)
                file2print_detail_train.flush()

                ACCs_run[run] = acc_test
                MAX_EPOCHs_run[run] = epoch_max

            ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
            MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

            print(f"{opt.dataset_name}\t{x_train.shape[0]}\t{x_test.shape[0]}\t{x_train.shape[1]}\t{opt.nb_class}"
                  f"\t{seed}\t{ACCs_seed[seed]}\t{MAX_EPOCHs_seed[seed]}", file=file2print_detail)
            file2print_detail.flush()

        ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
        ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
        MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

        print(f"{opt.dataset_name}\t{ACCs_seed_mean}\t{ACCs_seed_std}\t{MAX_EPOCHs_seed_max}", file=file2print)
        file2print.flush()