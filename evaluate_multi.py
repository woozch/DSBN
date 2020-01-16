import argparse
import logging
import os
import numpy as np
import pprint
import torch
import re

from torchvision import transforms
from dataset.factory import get_dataset
from model.factory import get_model
from torch.utils import data
from utils import io_utils, eval_utils
from dataset import datasets

import utils.custom_transforms as custom_transforms


def parse_args(args=None, namespace=None):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate DSBN Network. ' + \
                    'target label:0, sorce label:1,2,... \n' + \
                    '[digits: svhn, mnist, usps || ' + \
                    'office: amazon, webcam, dslr || ' + \
                    'office-home: Art, Clipart, Product, RealWorld || ' + \
                    'imageCLEF: caltech, pascal, imagenet || ' + \
                    'visDA: train, validation]')

    parser.add_argument('--model-name',
                        help="model name ['lenet', 'alexnet', 'resnet50', 'resnet50dsbn']",
                        default='resnet50', type=str)
    parser.add_argument('--exp-setting', help='exp setting[digits, office, imageclef, visda]', default='office',
                        type=str)
    parser.add_argument('--model-path', help='model path for evaluation', default='', type=str)
    parser.add_argument('--args-path', help='args path for evaluation. This code automatically ' +
                                            '"search base_dir/args_dict.pth" of model-path', default='', type=str)

    # model options
    parser.add_argument('--num-classes', help='number of classes', default=0, type=int)
    parser.add_argument('--source-datasets', help='source training dataset', default=['amazon', 'dslr'],
                        nargs='+', type=str)
    parser.add_argument('--target-dataset', help='target training dataset', default='webcam', type=str)
    parser.add_argument('--in-features', help='add in feature dimension. 0 for label logit space.', default=0, type=int)

    # machine options
    parser.add_argument('--num-workers', help='number of worker to load data', default=0, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=64, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    # log and diaplay
    parser.add_argument('--use-tfboard', help='whether use tensorflow tensorboard',
                        action='store_true')
    parser.add_argument('--print-console', help='activate console display', action='store_true')
    parser.add_argument('--save-results', help='whether to save the results', action='store_true')

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)  # set current gpu device id so pin_momory works on the target gpu
    if not os.path.isfile(args.model_path):
        raise IOError("ERROR model_path: {}".format(args.model_path))

    # load checkpoints
    checkpoint = torch.load(args.model_path)
    global_step = checkpoint['iteration']
    model_state_dict = checkpoint['model']

    # set logger
    model_dir = os.path.dirname(args.model_path)
    log_filename = 'evaluation_step{}.log'.format(global_step)
    log_path = os.path.join(model_dir, log_filename)
    logger = io_utils.get_logger(__name__, log_file=log_path, write_level=logging.INFO,
                                 print_level=logging.INFO if args.print_console else None)

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

    # update model args from filename
    model_args = io_utils.get_model_args_dict_from_filename(os.path.basename(args.model_path))
    model_args['source_datasets'] = model_args['source_dataset'].split('|')
    model_args['target_datasets'] = model_args['target_dataset'].split('|')
    args.__dict__.update(model_args)
    # load args if it exists
    args_path = os.path.join(model_dir, 'args_dict.pth')
    if os.path.isfile(args_path):
        logger.info('Arguemnt file exist. load arguments from {}'.format(args_path))
        args_dict = torch.load(args_path)
        update_dict = {'args_path': args_path,
                       'source_dataset': args_dict['source_dataset'],
                       'source_datasets': args_dict['source_datasets'],
                       'target_dataset': args_dict['target_dataset'],
                       'target_datasets': args_dict['target_datasets'],
                       'model_name': args_dict['model_name'],
                       'in_features': args_dict['in_features'], }
        args.__dict__.update(update_dict)
    args.dsbn = True if 'dsbn' in args.model_name else False  # set dsbn
    logger.info('\nArguments:\n' + pprint.pformat(vars(args), indent=4))

    model_options = io_utils.get_model_options_from_args(args, global_step)

    batch_size = args.batch_size
    num_classes = args.num_classes
    num_source_domains = len(args.source_datasets)
    num_target_domains = len(args.target_datasets)


    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        base_dir = os.path.dirname(args.model_path)
        tfboard_dir = os.path.join(base_dir, 'tfboard')
        if not os.path.isdir(tfboard_dir):
            os.makedirs(tfboard_dir)
        writer = SummaryWriter(tfboard_dir)
    ###################################################################################################################
    #                                               Data Loading                                                      #
    ###################################################################################################################

    source_test_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, source_dataset, 'test', args.jitter))
                            for source_dataset in args.source_datasets]
    target_test_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, target_dataset, 'test', args.jitter))
                            for target_dataset in args.target_datasets]

    ###################################################################################################################
    #                                               Model Loading                                                     #
    ###################################################################################################################
    model = get_model(args.model_name, args.num_classes, args.in_features, pretrained=False)

    logger.info('Load trained parameters...')
    model.load_state_dict(model_state_dict)
    model.train(False)
    model.eval()
    model = model.cuda(args.gpu)

    # tfboard: write centroids
    if args.use_tfboard:
        centroids_filename = io_utils.get_centroids_filename(model_options)
        centroids_path = os.path.join(model_dir, centroids_filename)
        if os.path.isfile(centroids_path):
            logger.info('write centroids on tfboard: {}'.format(centroids_path))
            centroids_ckpt = torch.load(centroids_path)

            for i, centroids in enumerate(centroids_ckpt['srcs_centroids']):
                src_centroids = centroids['centroids'].cpu().data.numpy()
                writer.add_embedding(src_centroids, metadata=list(range(num_classes)),
                                     tag='src_centroids_{}'.format(args.source_datasets[i]), global_step=global_step)

            trg_centroids = centroids_ckpt['trg_centroids']['centroids'].cpu().data.numpy()
            writer.add_embedding(trg_centroids, metadata=list(range(num_classes)),
                                 tag='trg_centroids', global_step=global_step)

    logger.info('Start Evaluation')
    results = {'step': global_step}
    total_features = []
    total_labels = []

    # for d_idx, dataset in enumerate(target_test_datasets + source_test_datasets):
    for d_idx, dataset in enumerate(target_test_datasets):
        # dataloader
        dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, drop_last=False, pin_memory=True)
        pred_vals = []
        y_vals = []
        if args.use_tfboard:
            features = []

        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(dataloader):
                x_val = x_val.cuda(args.gpu)
                y_val = y_val.cuda(args.gpu)

                if args.dsbn:
                    pred_val, f_val = model(x_val, torch.zeros_like(y_val), with_ft=True)
                else:
                    pred_val, f_val = model(x_val, with_ft=True)

                pred_vals.append(pred_val.cpu())
                y_vals.append(y_val.cpu())
                if args.use_tfboard:
                    features += [f_val.cpu().data.numpy()]

        pred_vals = torch.cat(pred_vals, 0)
        y_vals = torch.cat(y_vals, 0)
        test_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
        val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                       class_idx=c, topk=(1,))[0]))
                               for c, c_name in enumerate(dataset.classes)]
        # logging
        if d_idx <= num_target_domains:
            logger.info('{} Test Accuracy: {:.4f}%'.format(args.target_datasets[d_idx], 100 * test_accuracy))
            logger.info('\nEach class Accuracy of {}\n'.format(args.target_datasets[d_idx]) +
                        ''.join(["{:<25}: {:.2f}%\n".format(c_name, 100 * c_val_acc)
                                 for c_name, c_val_acc in val_accuracy_each_c]))
            logger.info('Evaluation mean Accuracy: {:.2f}%'.format(
                100 * float(torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))))
            if args.save_results:
                results.update({args.target_datasets[d_idx]: test_accuracy})
                results.update(
                    {args.target_datasets[d_idx] + '_' + c_name: c_val_acc for c_name, c_val_acc in val_accuracy_each_c})
        else:
            logger.info('{} Test Accuracy: {:.4f}'.format(args.source_datasets[d_idx - num_target_domains], test_accuracy))
            logger.info('\nEach class Accuracy of {}\n'.format(args.source_datasets[d_idx - num_target_domains]) +
                        ''.join(["{:<25}: {:.2f}%\n".format(c_name, 100 * c_val_acc)
                                 for c_name, c_val_acc in val_accuracy_each_c]))
            logger.info('Evaluation mean Accuracy: {:.2f}%'.format(
                100 * float(torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))))
            if args.save_results:
                results.update({args.source_datasets[d_idx-num_target_domains]: test_accuracy})
                results.update(
                    {args.source_datasets[d_idx - num_target_domains] + '_' + c_name: c_val_acc for c_name, c_val_acc in
                     val_accuracy_each_c})

        if args.use_tfboard:
            features = np.concatenate(features, axis=0)
            y_vals_numpy = y_vals.numpy().astype(np.int)
            embed_features = features
            # u, s, vt = np.linalg.svd(features)
            # embed_features = np.dot(features, vt[:3, :].transpose())

            if d_idx <= num_target_domains:
                total_features += [embed_features]
                total_labels += [args.target_datasets[d_idx][0] + str(int(l)) for l in y_vals]
                writer.add_embedding(embed_features, metadata=y_vals_numpy, tag=args.target_datasets[d_idx],
                                     global_step=global_step)
            else:
                total_features += [embed_features]
                total_labels += [args.source_datasets[d_idx-num_target_domains][0] + str(int(l)) for l in y_vals]
                writer.add_embedding(embed_features, metadata=y_vals_numpy, tag=args.source_datasets[d_idx - num_target_domains],
                                     global_step=global_step)

    if args.use_tfboard:
        total_features = np.concatenate(total_features, axis=0)
        writer.add_embedding(total_features, metadata=list(total_labels),
                             tag='feat_embed_S:{}_T:{}'.format(args.source_dataset, args.target_dataset),
                             global_step=global_step)

    # save results
    if args.save_results:
        result_filename = 'evaluation_{:06d}.pth'.format(global_step)
        torch.save(results, os.path.join(model_dir, result_filename))

    if args.use_tfboard:
        writer.close()


if __name__ == '__main__':
    main()
