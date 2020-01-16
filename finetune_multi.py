import inits
import argparse
import logging
import pprint
import datetime
import sys
import random

from dataset.factory import get_dataset
from model.factory import get_model
from discriminator.factory import get_discriminator

from model.centroids import Centroids
from torch.utils import data
from utils.train_utils import adaptation_factor, semantic_loss_calc, L2_u_p_loss, KL_u_p_loss, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils.train_utils import one_hot_encoding
from utils import io_utils, eval_utils

import torch.nn.functional as F
import torch.optim as optim
import os, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# TODO: cpua implementation not included in this version
def parse_args(args=None, namespace=None):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Finetune DSBN Network.\n' + \
                    'target label:0, sorce label:1,2,... \n' + \
                    '[digits: svhn, mnist, usps || ' + \
                    'office: amazon, webcam, dslr || ' + \
                    'office-home: Art, Clipart, Product, RealWorld || ' + \
                    'imageCLEF: caltech, pascal, imagenet || ' + \
                    'visDA: train, validation]')
    parser.add_argument('--model-name',
                        help="model name ['lenet', 'resnet50', resnet50dsbn', 'resnet101', resnet101dsbn']",

                        default='resnet50', type=str)
    parser.add_argument('--exp-setting', help='exp setting[digits, office, imageclef, visda]', default='office',
                        type=str)
    parser.add_argument('--teacher-model-path', help='teacher model path', default='', type=str)
    parser.add_argument('--init-model-path', help='init model path', default='', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='output/office_default', type=str)

    # model options
    parser.add_argument('--num-classes', help='number of classes', default=0, type=int)
    parser.add_argument('--source-datasets', help='source training dataset', default=['amazon', 'dslr'],
                        nargs='+')
    parser.add_argument('--merge-sources', help='Use merged dataset as source dataset.', action='store_true')
    parser.add_argument('--target-datasets', help='target training dataset', default=['webcam'], nargs='+')
    parser.add_argument('--in-features', help='add in feature dimension. 0 for label logit space.', default=0,
                        type=int)
    parser.add_argument('--jitter', default='None', type=str,
                        help='data loader additional jitter type. None option is default jittering(rgb224)' +
                             ' [None, grey224, grey160, rgb160]')

    # machine options
    parser.add_argument('--num-workers', help='number of worker to load data', default=2, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument('--manual-seed', type=int, default=0, help='manual random seed')

    # train hyper-parameters
    parser.add_argument('--max-step', help='maximum step', default=50000, type=int)
    parser.add_argument('--early-stop-step', help='early stop step', default=30000, type=int)
    parser.add_argument('--warmup-learning-rate', '-wlr', help='warmup learning rate', default=0.0, type=float)
    parser.add_argument('--warmup-step', type=int, default=0, help='warm-up iterations')
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=5e-5, type=float)
    parser.add_argument('--beta1', dest='beta1', help='beta1 for Adam', default=0.9, type=float)
    parser.add_argument('--beta2', dest='beta2', help='beta2 for Adam', default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--double-bias-lr', help='double-bias', action='store_true')
    parser.add_argument('--base-weight-factor', help='reduce base_weight learning rate by the factor value',
                        default=0.1, type=float)
    parser.add_argument('--sm-etha', help='sm loss adjust factor', default=1.0, type=float)
    parser.add_argument('--pseudo-target-threshold', help='threshold for calculating pseudo-t loss', default=0.0,
                        type=float)
    parser.add_argument('--no-lambda', help='double-bias', action='store_true')
    parser.add_argument('--optimizer', help='[Adam/SGD]', default='Adam', type=str)

    # trainval parameters
    parser.add_argument('--adaptation-gamma', help='adaptation gamma value', default=10, type=float)
    parser.add_argument('--domain-loss-adjust-factor', help='domain loss factor', default=0.1, type=float)
    parser.add_argument('--adv-loss', help='add domain loss', action='store_true')
    parser.add_argument('--sm-loss', help='add moving semantic loss', action='store_true')
    parser.add_argument('--pseudo-target-loss',
                        help='target classification loss with pseudo label, [default/score]_[ensemble]_[fix]',
                        default='', type=str)

    # log and diaplay
    parser.add_argument('--use-tfboard', help='whether use tensorflow tensorboard',
                        action='store_true')
    parser.add_argument('--save-model-hist', help='save model histogram on tfboard', action='store_true')
    parser.add_argument('--disp-interval', help='number of iterations to display', default=10,
                        type=int)
    parser.add_argument('--save-interval',
                        help='number of iterations to save. if save_interval < 0, no saving mode.',
                        default=500,
                        type=int)
    parser.add_argument('--print-console', help='activate console display', action='store_true')
    parser.add_argument('--save-ckpts', help='whether to save the checkpoints in every save_interval',
                        action='store_true')
    parser.add_argument('--resume', help='resume from latest(or best) checkpoint', action='store_true')

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    args.dsbn = True if 'dsbn' in args.model_name else False  # set dsbn
    args.cpua = True if 'cpua' in args.model_name else False
    args.source_dataset = '|'.join(args.source_datasets)
    args.target_dataset = '|'.join(args.target_datasets)
    torch.cuda.set_device(args.gpu)  # set current gpu device id so pin_momory works on the target gpu
    start_time = datetime.datetime.now()  # execution start time

    # make save_dir
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # check whether teacher model exists
    if not os.path.isfile(args.teacher_model_path):
        raise AttributeError('Missing teacher model path: {}'.format(args.teacher_model_path))

    # create log file
    log_filename = 'train_records.log'
    log_path = os.path.join(args.save_dir, log_filename)
    logger = io_utils.get_logger(__name__, log_file=log_path, write_level=logging.INFO,
                                 print_level=logging.INFO if args.print_console else None,
                                 mode='a' if args.resume else 'w')

    # set num_classes by checking exp_setting
    if args.num_classes == 0:
        if args.exp_setting == 'digits':
            logger.warning('num_classes are not 10! set to 10.')
            args.num_classes = 10
        elif args.exp_setting == 'office':
            logger.warning('num_classes are not 31! set to 31.')
            args.num_classes = 31
        elif args.exp_setting in ['visda', 'imageclef']:
            logger.warning('num_classes are not 12! set to 12.')
            args.num_classes = 12
        elif args.exp_setting in ['office-home']:
            logger.warning('num_classes are not 65! set to 65.')
            args.num_classes = 65
        elif args.exp_setting in ['office-caltech']:
            args.num_classes = 10
        else:
            raise AttributeError('Wrong num_classes: {}'.format(args.num_classes))

    if args.manual_seed:
        # set manual seed
        args.manual_seed = np.uint32(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        logger.info('Random Seed: {}'.format(int(args.manual_seed)))
        args.random_seed = args.manual_seed  # save seed into args
    else:
        seed = np.uint32(random.randrange(sys.maxsize))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(np.uint32(seed))
        logger.info('Random Seed: {}'.format(seed))
        args.random_seed = seed  # save seed into args

    if args.resume:
        logger.info('Resume training')
    else:
        logger.info('\nArguments:\n' + pprint.pformat(vars(args), indent=4))  # print args
    torch.save(vars(args), os.path.join(args.save_dir, 'args_dict.pth'))  # save args

    num_classes = args.num_classes
    in_features = args.in_features if args.in_features != 0 else num_classes
    num_domains = len(args.source_datasets) + len(args.target_datasets)
    if args.merge_sources:
        num_source_domains = 1
    else:
        num_source_domains = len(args.source_datasets)
    num_target_domains = len(args.target_datasets)

    # tfboard
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        tfboard_dir = os.path.join(args.save_dir, 'tfboard')
        if not os.path.isdir(tfboard_dir):
            os.makedirs(tfboard_dir)
        writer = SummaryWriter(tfboard_dir)

    # resume
    if args.resume:
        try:
            checkpoints = io_utils.load_latest_checkpoints(args.save_dir, args, logger)
        except FileNotFoundError:
            logger.warning('Latest checkpoints are not found! Trying to load best model...')
            checkpoints = io_utils.load_best_checkpoints(args.save_dir, args, logger)

        start_iter = checkpoints[0]['iteration'] + 1
    else:
        start_iter = 1

    ###################################################################################################################
    #                                               Data Loading                                                      #
    ###################################################################################################################

    source_train_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, source_name, 'train', args.jitter))
                             for source_name in args.source_datasets]
    target_train_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, target_name, 'train', args.jitter))
                             for target_name in args.target_datasets]

    if args.merge_sources:
        for i in range(len(source_train_datasets)):
            if i == 0:
                merged_source_train_datasets = source_train_datasets[i]
            else:
                # concatenate dataset
                merged_source_train_datasets = merged_source_train_datasets + source_train_datasets[i]
        source_train_datasets = [merged_source_train_datasets]

    # dataloader
    source_train_dataloaders = [data.DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
                                for source_train_dataset in source_train_datasets]
    target_train_dataloaders = [data.DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
                                for target_train_dataset in target_train_datasets]

    source_train_dataloader_iters = [enumerate(source_train_dataloader) for source_train_dataloader in
                                     source_train_dataloaders]
    target_train_dataloader_iters = [enumerate(target_train_dataloader) for target_train_dataloader in
                                     target_train_dataloaders]

    # validation dataloader
    target_val_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, target_name, 'val', args.jitter))
                           for target_name in args.target_datasets]
    target_val_dataloaders = [data.DataLoader(target_val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
                              for target_val_dataset in target_val_datasets]

    ###################################################################################################################
    #                                               Model Loading                                                     #
    ###################################################################################################################
    model = get_model(args.model_name, args.num_classes, args.in_features, num_domains=num_domains, pretrained=True)

    model.train(True)
    if args.resume:
        model.load_state_dict(checkpoints[0]['model'])
    elif args.init_model_path:
        init_checkpoint = torch.load(args.init_model_path)
        model.load_state_dict(init_checkpoint['model'])
    model = model.cuda(args.gpu)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=args.double_bias_lr, base_weight_factor=args.base_weight_factor)

    if args.adv_loss:
        if args.cpua:
            discriminators = [get_discriminator('cpua',
                                                in_features=args.in_features if args.in_features != 0 else args.num_classes)
                              for _ in range(num_target_domains) for _ in range(num_source_domains)]
        else:
            discriminators = [get_discriminator(args.exp_setting,
                                                in_features=args.in_features if args.in_features != 0 else args.num_classes)
                              for _ in range(num_target_domains) for _ in range(num_source_domains)]
        discriminators = [discriminator.cuda(args.gpu) for discriminator in discriminators]
        D_params = get_optimizer_params(discriminators, args.learning_rate, weight_decay=args.weight_decay,
                                        double_bias_lr=args.double_bias_lr, base_weight_factor=None)
        if args.resume:
            if checkpoints[1]:
                for d_idx, discriminator in enumerate(discriminators):
                    discriminator.load_state_dict(checkpoints[1]['discriminators'][d_idx])

    if args.sm_loss:
        srcs_centroids = [Centroids(in_features, num_classes) for _ in range(num_source_domains)]
        trgs_centroids = [Centroids(in_features, num_classes) for _ in range(num_target_domains)]

        if args.resume:
            if checkpoints[2]:
                for src_idx, src_centroids in enumerate(srcs_centroids):
                    src_centroids.load_state_dict(checkpoints[2]['srcs_centroids'][src_idx])
                for trg_idx, trg_centroids in enumerate(trgs_centroids):
                    trg_centroids.load_state_dict(checkpoints[2]['trgs_centroids'][trg_idx])

        srcs_centroids = [src_centroids.cuda(args.gpu) for src_centroids in srcs_centroids]
        trgs_centroids = [trg_centroids.cuda(args.gpu) for trg_centroids in trgs_centroids]

    # teacher model
    teacher_model_args = io_utils.get_model_args_dict_from_filename(os.path.basename(args.teacher_model_path))
    args.teacher_dsbn = True if 'dsbn' in teacher_model_args['model_name'] else False
    teacher_model = get_model(teacher_model_args['model_name'], args.num_classes, teacher_model_args['in_features'],
                              num_domains=num_domains, pretrained=False)
    teacher_model.load_state_dict(torch.load(args.teacher_model_path)['model'])
    teacher_model.train(False)
    teacher_model.eval()
    teacher_model = teacher_model.cuda(args.gpu)

    ###################################################################################################################
    #                                               Train Configurations                                              #
    ###################################################################################################################
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    # mse_loss = nn.MSELoss()

    lr_scheduler = LRScheduler(args.learning_rate, args.warmup_learning_rate, args.warmup_step,
                               num_steps=args.max_step,
                               alpha=10, beta=0.75, double_bias_lr=args.double_bias_lr,
                               base_weight_factor=args.base_weight_factor)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(params, betas=(args.beta1, args.beta2))

    if args.resume:
        if checkpoints[1]:
            optimizer.load_state_dict(checkpoints[1]['optimizer'])

    if args.adv_loss:
        if args.optimizer.lower() == 'sgd':
            optimizer_D = optim.SGD(D_params, momentum=0.9, nesterov=True)
        else:
            optimizer_D = optim.Adam(D_params, betas=(args.beta1, args.beta2))

        if args.resume:
            if checkpoints[1]:
                optimizer_D.load_state_dict(checkpoints[1]['optimizer_D'])

    # Train Starts
    logger.info('Train Starts')
    domain_loss_adjust_factor = args.domain_loss_adjust_factor

    monitor = Monitor()

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies
    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i_iter in range(start_iter, args.early_stop_step + 1):
        src_inputs = []
        for src_dataloader_idx in range(len(source_train_dataloader_iters)):
            try:
                _, (x_s, y_s) = source_train_dataloader_iters[src_dataloader_idx].__next__()
                src_inputs.append((x_s, y_s))
            except StopIteration:
                source_train_dataloader_iters[src_dataloader_idx] = enumerate(
                    source_train_dataloaders[src_dataloader_idx])
                _, (x_s, y_s) = source_train_dataloader_iters[src_dataloader_idx].__next__()
                src_inputs.append((x_s, y_s))

        trg_inputs = []
        for trg_dataloader_idx in range(len(target_train_dataloader_iters)):
            try:
                _, (x_t, _) = target_train_dataloader_iters[trg_dataloader_idx].__next__()
                trg_inputs.append((x_t, None))
            except StopIteration:
                target_train_dataloader_iters[trg_dataloader_idx] = enumerate(
                    target_train_dataloaders[trg_dataloader_idx])
                _, (x_t, _) = target_train_dataloader_iters[trg_dataloader_idx].__next__()
                trg_inputs.append((x_t, None))

        current_lr = lr_scheduler.current_lr(i_iter)
        adaptation_lambda = adaptation_factor((i_iter - args.warmup_step) / float(args.max_step),
                                              gamma=args.adaptation_gamma)
        if args.no_lambda:
            adaptation_lambda = 1.0

        # init optimizer
        optimizer.zero_grad()
        lr_scheduler(optimizer, i_iter)
        if args.adv_loss:
            optimizer_D.zero_grad()
            lr_scheduler(optimizer_D, i_iter)

        ########################################################################################################
        #                                               Train G                                                #
        ########################################################################################################
        if args.adv_loss:
            for discriminator in discriminators:
                for param in discriminator.parameters():
                    param.requires_grad = False
        # ship to cuda
        src_inputs = [(x_s.cuda(args.gpu), y_s.cuda(args.gpu)) for (x_s, y_s) in src_inputs]
        trg_inputs = [(x_t.cuda(args.gpu), None) for (x_t, _) in trg_inputs]

        if args.dsbn:
            src_preds = []
            for src_idx, (x_s, y_s) in enumerate(src_inputs):
                pred_s, f_s = model(x_s, src_idx * torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu),
                                    with_ft=True)
                src_preds.append((pred_s, f_s))

            trg_preds = []
            for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                pred_t, f_t = model(x_t, trg_idx * torch.ones(x_t.shape[0], dtype=torch.long).cuda(args.gpu),
                                    with_ft=True)
                trg_preds.append((pred_t, f_t))

        else:
            src_preds = []
            for src_idx, (x_s, y_s) in enumerate(src_inputs):
                pred_s, f_s = model(x_s, with_ft=True)
                src_preds.append((pred_s, f_s))

            trg_preds = []
            for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                pred_t, f_t = model(x_t, with_ft=True)
                trg_preds.append((pred_t, f_t))

        if args.sm_loss or args.pseudo_target_loss or args.class_balance_loss:
            
            with torch.no_grad():
                if args.teacher_dsbn:
                    preds_t_pseudos = []
                    for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                        pred_t_pseudo = teacher_model(x_t, trg_idx * torch.ones(x_t.shape[0], dtype=torch.long).cuda(
                            args.gpu),
                                                      with_ft=False)
                        preds_t_pseudos.append(pred_t_pseudo)
                else:
                    preds_t_pseudos = []
                    for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                        pred_t_pseudo = teacher_model(x_t, with_ft=False)
                        preds_t_pseudos.append(pred_t_pseudo)

            if 'ensemble' in args.pseudo_target_loss:
                if args.dsbn:
                    with torch.no_grad():
                        model.eval()
                        preds_t_pseudos_student = []
                        for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                            pred_t_pseudo_student = model(x_t,
                                                          trg_idx * torch.ones(x_t.shape[0], dtype=torch.long).cuda(
                                                              args.gpu),
                                                          with_ft=False)
                            preds_t_pseudos_student.append(pred_t_pseudo_student)
                        model.train(True)
                else:
                    with torch.no_grad():
                        model.eval()
                        preds_t_pseudos_student = []
                        for trg_idx, (x_t, _) in enumerate(trg_inputs, num_source_domains):
                            pred_t_pseudo_student = model(x_t, with_ft=False)
                            preds_t_pseudos_student.append(pred_t_pseudo_student)
                        model.train(True)
                if 'ensemble_fix' in args.pseudo_target_loss:
                    preds_t_pseudos = [pred_t_pseudo + pred_t_pseudo_student / 2 for
                                       pred_t_pseudo, pred_t_pseudo_student in
                                       zip(preds_t_pseudos, preds_t_pseudos_student)]
                else:
                    pseudo_label_lambda = adaptation_factor((i_iter - args.warmup_step) / float(args.max_step),
                                                            gamma=args.adaptation_gamma)
                    monitor.update({"pseudo_lambda": pseudo_label_lambda})
                    preds_t_pseudos = [pred_t_pseudo * (
                            1 - pseudo_label_lambda) + pred_t_pseudo_student * pseudo_label_lambda for
                                       pred_t_pseudo, pred_t_pseudo_student in
                                       zip(preds_t_pseudos, preds_t_pseudos_student)]

        Closs_src = 0
        for (_, y_s), (pred_s, _) in zip(src_inputs, src_preds):
            Closs_src = Closs_src + ce_loss(pred_s, y_s) / float(num_source_domains)

        monitor.update({"Loss/Closs_src": float(Closs_src)})
        Floss = Closs_src

        if args.pseudo_target_loss:
            for (pred_t, f_t), pred_t_pseudo in zip(trg_preds, preds_t_pseudos):
                score_t_pseudo, y_t_pseudo = torch.max(F.softmax(pred_t_pseudo, 1), 1)
                selected_y_i = score_t_pseudo >= args.pseudo_target_threshold
                if len(pred_t[selected_y_i]) > 0:
                    if 'default' in args.pseudo_target_loss:
                        Closs_trg = ce_loss(pred_t[selected_y_i], y_t_pseudo[selected_y_i])
                        Floss = Floss + adaptation_lambda * Closs_trg
                    elif 'score' in args.pseudo_target_loss:
                        weighted_pred_t = pred_t[selected_y_i] * score_t_pseudo[selected_y_i].unsqueeze(1)
                        Closs_trg = ce_loss(weighted_pred_t, y_t_pseudo[selected_y_i])
                        Floss = Floss + Closs_trg
                    monitor.update(
                        {"Score/trg_ps": float(
                            torch.mean(score_t_pseudo[score_t_pseudo >= args.pseudo_target_threshold]))})
                    monitor.update({"Loss/Closs_trg": float(Closs_trg)})

        if args.adv_loss:
            # adversarial loss
            Gloss = 0
            for trg_idx, (_, f_t) in enumerate(trg_preds):
                for src_idx, (_, f_s) in enumerate(src_preds):
                    Dout_s = discriminators[trg_idx * num_source_domains + src_idx](f_s)
                    source_label = torch.zeros_like(Dout_s).cuda(args.gpu)
                    loss_adv_src = domain_loss_adjust_factor * bce_loss(Dout_s, source_label) / 2

                    Dout_t = discriminators[trg_idx * num_source_domains + src_idx](f_t)
                    target_label = torch.ones_like(Dout_t).cuda(args.gpu)
                    loss_adv_trg = domain_loss_adjust_factor * bce_loss(Dout_t, target_label) / 2
                    Gloss = Gloss - (loss_adv_src + loss_adv_trg)
            Gloss = Gloss / float(num_target_domains * num_source_domains)
            monitor.update({'Loss/Gloss': float(Gloss)})

            Floss = Floss + adaptation_lambda * Gloss

        # moving semantic loss
        if args.sm_loss:
            current_srcs_centroids = [src_centroids(f_s, y_s) for src_centroids, (x_s, y_s), (_, f_s) in
                                      zip(srcs_centroids, src_inputs, src_preds)]

            current_trgs_centroids = [trg_centroids(f_t, torch.argmax(pred_t_pseudo, 1)) for
                                      trg_centroids, pred_t_pseudo, (_, f_t) in
                                      zip(trgs_centroids, preds_t_pseudos, trg_preds)]

            semantic_loss = 0
            for current_trg_centroids in current_trgs_centroids:
                for current_src_centroids in current_srcs_centroids:
                    semantic_loss = semantic_loss + args.sm_etha * semantic_loss_calc(current_src_centroids,
                                                                                      current_trg_centroids)
            semantic_loss = semantic_loss / float(num_target_domains * num_source_domains)
            monitor.update({'Loss/SMloss': float(semantic_loss)})

            Floss = Floss + adaptation_lambda * semantic_loss

        # Floss backward
        Floss.backward()
        optimizer.step()
        ########################################################################################################
        #                                               Train D                                                #
        ########################################################################################################
        if args.adv_loss:
            for discriminator in discriminators:
                for param in discriminator.parameters():
                    param.requires_grad = True

        if args.adv_loss:
            # adversarial loss
            Dloss = 0
            for trg_idx, (_, f_t) in enumerate(trg_preds):
                for src_idx, (_, f_s) in enumerate(src_preds):
                    Dout_s = discriminators[trg_idx * num_source_domains + src_idx](f_s.detach())
                    source_label = torch.zeros_like(Dout_s).cuda(args.gpu)
                    loss_adv_src = domain_loss_adjust_factor * bce_loss(Dout_s, source_label) / 2

                    # target
                    Dout_t = discriminators[trg_idx * num_source_domains + src_idx](f_t.detach())
                    target_label = torch.ones_like(Dout_t).cuda(args.gpu)
                    loss_adv_trg = domain_loss_adjust_factor * bce_loss(Dout_t, target_label) / 2
                    Dloss = Dloss + loss_adv_src + loss_adv_trg
            Dloss = Dloss / float(num_target_domains * num_source_domains)
            monitor.update({'Loss/Dloss': float(Dloss)})
            Dloss = adaptation_lambda * Dloss
            Dloss.backward()
            optimizer_D.step()

        if args.sm_loss:
            for src_centroids, current_src_centroids in zip(srcs_centroids, current_srcs_centroids):
                src_centroids.centroids.data = current_src_centroids.data
            for trg_centroids, current_trg_centroids in zip(trgs_centroids, current_trgs_centroids):
                trg_centroids.centroids.data = current_trg_centroids.data

        if i_iter % args.disp_interval == 0 and i_iter != 0:
            disp_msg = 'iter[{:8d}/{:8d}], '.format(i_iter, args.early_stop_step)
            disp_msg += str(monitor)
            if args.adv_loss or args.sm_loss:
                disp_msg += ', lambda={:.6f}'.format(adaptation_lambda)
            disp_msg += ', lr={:.6f}'.format(current_lr)
            logger.info(disp_msg)

            if args.use_tfboard:
                if args.save_model_hist:
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.cpu().data.numpy(), i_iter, bins='auto')

                for k, v in monitor.losses.items():
                    writer.add_scalar(k, v, i_iter)
                if args.adv_loss or args.sm_loss:
                    writer.add_scalar('adaptation_lambda', adaptation_lambda, i_iter)
                writer.add_scalar('learning rate', current_lr, i_iter)
            monitor.reset()

        if i_iter % args.save_interval == 0 and i_iter != 0:
            logger.info("Elapsed Time: {}".format(datetime.datetime.now() - start_time))
            logger.info("Start Evaluation at {:d}".format(i_iter))

            target_val_dataloader_iters = [enumerate(target_val_dataloader)
                                           for target_val_dataloader in target_val_dataloaders]

            total_val_accuracies = []
            mean_val_accuracies = []
            val_accuracies_each_c = []
            model.eval()  # evaluation mode
            for trg_idx, target_val_dataloader_iter in enumerate(target_val_dataloader_iters, num_source_domains):
                pred_vals = []
                y_vals = []
                x_val = None
                y_val = None
                pred_val = None
                with torch.no_grad():
                    for i, (x_val, y_val) in target_val_dataloader_iter:
                        y_vals.append(y_val.cpu())
                        x_val = x_val.cuda(args.gpu)
                        y_val = y_val.cuda(args.gpu)

                        if args.dsbn:
                            pred_val = model(x_val, trg_idx * torch.ones_like(y_val), with_ft=False)
                        else:
                            pred_val = model(x_val, with_ft=False)

                        pred_vals.append(pred_val.cpu())

                pred_vals = torch.cat(pred_vals, 0)
                y_vals = torch.cat(y_vals, 0)
                total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])

                val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                               class_idx=c, topk=(1,))[0]))
                                       for c, c_name in
                                       enumerate(target_val_datasets[trg_idx - num_source_domains].classes)]
                logger.info('\n{} Accuracy of Each class\n'.format(args.target_datasets[trg_idx - num_source_domains]) +
                            ''.join(["{:<25}: {:.2f}%\n".format(c_name, 100 * c_val_acc)
                                     for c_name, c_val_acc in val_accuracy_each_c]))
                mean_val_accuracy = float(
                    torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))

                logger.info('{} mean Accuracy: {:.2f}%'.format(
                    args.target_datasets[trg_idx - num_source_domains], 100 * mean_val_accuracy))
                logger.info(
                    '{} Accuracy: {:.2f}%'.format(args.target_datasets[trg_idx - num_source_domains],
                                                  total_val_accuracy * 100))

                total_val_accuracies.append(total_val_accuracy)
                val_accuracies_each_c.append(val_accuracy_each_c)
                mean_val_accuracies.append(mean_val_accuracy)

                if args.use_tfboard:
                    writer.add_scalar('Val_acc', total_val_accuracy, i_iter)
                    for c_name, c_val_acc in val_accuracy_each_c:
                        writer.add_scalar('Val_acc_of_{}'.format(c_name), c_val_acc)
            model.train(True)  # train mode

            if args.exp_setting.lower() == 'visda':
                val_accuracy = float(torch.mean(torch.FloatTensor(mean_val_accuracies)))
            else:
                val_accuracy = float(torch.mean(torch.FloatTensor(total_val_accuracies)))

            # for memory
            del x_val, y_val, pred_val, pred_vals, y_vals
            for target_val_dataloader_iter in target_val_dataloader_iters:
                del target_val_dataloader_iter
            del target_val_dataloader_iters

            if val_accuracy > best_accuracy:
                # save best model
                best_accuracy = val_accuracy
                best_accuracies_each_c = val_accuracies_each_c
                best_mean_val_accuracies = mean_val_accuracies
                best_total_val_accuracies = total_val_accuracies
                options = io_utils.get_model_options_from_args(args, i_iter)
                # dict to save
                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}
                if args.adv_loss:
                    optimizer_dict.update({'optimizer_D': optimizer_D.state_dict(),
                                           'discriminators': [discriminator.cpu().state_dict()
                                                              for discriminator in discriminators],
                                           'source_datasets': args.source_datasets,
                                           'target_datasets': args.target_datasets})
                centroids_dict = {}
                if args.sm_loss:
                    centroids_dict = {
                        'srcs_centroids': [src_centroids.cpu().state_dict() for src_centroids in srcs_centroids],
                        'trgs_centroids': [trg_centroids.cpu().state_dict() for trg_centroids in trgs_centroids]}
                # save best checkpoint
                io_utils.save_checkpoints(args.save_dir, options, i_iter, model_dict, optimizer_dict, centroids_dict,
                                          logger, best=True)
                # ship to cuda
                model = model.cuda(args.gpu)
                if args.adv_loss:
                    discriminators = [discriminator.cuda(args.gpu) for discriminator in discriminators]
                if args.sm_loss:
                    srcs_centroids = [src_centroids.cuda(args.gpu) for src_centroids in srcs_centroids]
                    trgs_centroids = [trg_centroids.cuda(args.gpu) for trg_centroids in trgs_centroids]

                # save best result into textfile
                contents = [' '.join(sys.argv) + '\n',
                            "best accuracy: {:.2f}%\n".format(best_accuracy)]
                for d_idx in range(num_target_domains):
                    best_accuracy_each_c = best_accuracies_each_c[d_idx]
                    best_mean_val_accuracy = best_mean_val_accuracies[d_idx]
                    best_total_val_accuracy = best_total_val_accuracies[d_idx]
                    contents.extend(["{}2{}\n".format(args.source_dataset, args.target_datasets[d_idx]),
                                     "best total acc: {:.2f}%\n".format(100 * best_total_val_accuracy),
                                     "best mean acc: {:.2f}%\n".format(100 * best_mean_val_accuracy),
                                     'Best Accs: ' + ''.join(["{:.2f}% ".format(100 * c_val_acc)
                                                              for _, c_val_acc in best_accuracy_each_c]) + '\n'])

                best_result_path = os.path.join('./output', '{}_best_result.txt'.format(
                    os.path.splitext(os.path.basename(__file__))[0]))
                with open(best_result_path, 'a+') as f:
                    f.writelines(contents)

            # logging best model results
            for trg_idx in range(num_target_domains):
                best_accuracy_each_c = best_accuracies_each_c[trg_idx]
                best_total_val_accuracy = best_total_val_accuracies[trg_idx]
                best_mean_val_accuracy = best_mean_val_accuracies[trg_idx]
                logger.info(
                    '\nBest {} Accuracy of Each class\n'.format(args.target_datasets[trg_idx]) +
                    ''.join(["{:<25}: {:.2f}%\n".format(c_name, 100 * c_val_acc)
                             for c_name, c_val_acc in best_accuracy_each_c]))
                logger.info('Best Accs: ' + ''.join(["{:.2f}% ".format(100 * c_val_acc)
                                                     for _, c_val_acc in best_accuracy_each_c]))
                logger.info('Best {} mean Accuracy: {:.2f}%'.format(args.target_datasets[trg_idx],
                                                                    100 * best_mean_val_accuracy))
                logger.info('Best {} Accuracy: {:.2f}%'.format(args.target_datasets[trg_idx],
                                                               100 * best_total_val_accuracy))
            logger.info("Best model's Average Accuracy of targets: {:.2f}".format(100 * best_accuracy))

            if args.save_ckpts:
                # get options
                options = io_utils.get_model_options_from_args(args, i_iter)
                # dict to save
                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}
                if args.adv_loss:
                    optimizer_dict.update({'optimizer_D': optimizer_D.state_dict(),
                                           'discriminators': [discriminator.cpu().state_dict()
                                                              for discriminator in discriminators]})
                centroids_dict = {}
                if args.sm_loss:
                    centroids_dict = {
                        'srcs_centroids': [src_centroids.cpu().state_dict() for src_centroids in srcs_centroids],
                        'trgs_centroids': [trg_centroids.cpu().state_dict() for trg_centroids in trgs_centroids]}
                # save checkpoint
                io_utils.save_checkpoints(args.save_dir, options, i_iter, model_dict, optimizer_dict, centroids_dict,
                                          logger, best=False)

                # ship to cuda
                model = model.cuda(args.gpu)
                if args.adv_loss:
                    discriminators = [discriminator.cuda(args.gpu) for discriminator in discriminators]
                if args.sm_loss:
                    srcs_centroids = [src_centroids.cuda(args.gpu) for src_centroids in srcs_centroids]
                    trgs_centroids = [trg_centroids.cuda(args.gpu) for trg_centroids in trgs_centroids]

    if args.use_tfboard:
        writer.close()

    logger.info('Total Time: {}'.format((datetime.datetime.now() - start_time)))


if __name__ == '__main__':
    main()
