import os
import os.path
import hashlib
import errno
import logging
from collections import defaultdict
from string import Formatter
import torch
import re


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def get_logger(name, fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s', print_level=logging.DEBUG,
               write_level=logging.DEBUG, log_file='', mode='w'):
    """
    Get Logger with given name
    :param name: logger name.
    :param fmt: log format. (default: %(asctime)s:%(levelname)s:%(name)s:%(message)s)
    :param level: logging level. (default: logging.DEBUG)
    :param log_file: path of log file. (default: None)
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(write_level)
    formatter = logging.Formatter(fmt)
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(write_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if print_level is not None:
        try:
            import coloredlogs
            coloredlogs.install(level=print_level, logger=logger)
        except ImportError:
            # Add stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(print_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger


def get_directory_template(args_dict):
    model_options = get_model_options_from_args(args_dict)
    if not isinstance(model_options, defaultdict):
        temp = defaultdict(lambda: 'None')
        temp.update(model_options)
        model_options = temp

    model_dir = "{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}" + \
                "_{max_step}_{early_stop_step}_{warmup_learning_rate}_{warmup_step}_{learning_rate}_{beta1}_{beta2}" + \
                "_{weight_decay}_{double_bias_lr}_{base_weight_factor}" + \
                "_{adaptation_gamma}_{domain_loss_adjust_factor}" + \
                "_{adv_loss}_{sm_loss}+{sm_etha}_{pseudo_target_loss}" + \
                "_{knowledge_distillation_alpha}_{knowledge_distillation_temperature}"

    fmtr = Formatter()
    return fmtr.vformat(model_dir, (), model_options)


def get_model_filename(model_options, best=False):
    if not isinstance(model_options, defaultdict):
        temp = defaultdict(lambda: 'None')
        temp.update(model_options)
        model_options = temp
    if best:
        file_name = "best_{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}.pth"
    else:
        file_name = "{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}_{i_iter:06d}.pth"
    fmtr = Formatter()
    return fmtr.vformat(file_name, (), model_options)


def get_model_args_dict_from_filename(model_file_name):
    pattern = r'^(best_|)(?P<model_name>.+)\+(?P<jitter>.+)\+i(?P<in_features>.+)_(?P<source_dataset>[A-Za-z\|]+)2' + \
              '(?P<target_dataset>[A-Za-z\|]+)(_(?P<i_iter>[\d]{6})|)\.pth$'
    search = re.search(pattern, model_file_name)
    if not search:
        raise AttributeError('Cannot parse model_args from {}'.format(model_file_name))

    i_iter = int(search.group("i_iter")) if search.group("i_iter") else 0

    model_args_dict = {
        'model_name': str(search.group("model_name")),
        'jitter': str(search.group("jitter")),
        'in_features': int(search.group("in_features")),
        'source_dataset': str(search.group("source_dataset")),
        'target_dataset': str(search.group("target_dataset")),
        'i_iter': i_iter
    }
    return model_args_dict


def get_optimizer_filename(model_options, best=False):
    if not isinstance(model_options, defaultdict):
        temp = defaultdict(lambda: 'None')
        temp.update(model_options)
        model_options = temp
    if best:
        file_name = "best_optimizer_{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}.pth"
    else:
        file_name = "optimizer_{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}_{i_iter:06d}.pth"
    fmtr = Formatter()
    return fmtr.vformat(file_name, (), model_options)


def get_centroids_filename(model_options, best=False):
    if not isinstance(model_options, defaultdict):
        temp = defaultdict(lambda: 'None')
        temp.update(model_options)
        model_options = temp
    if best:
        file_name = "best_centroids_{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}.pth"
    else:
        file_name = "centroids_{model_name}+{jitter}+i{in_features}_{source_dataset}2{target_dataset}_{i_iter:06d}.pth"
    fmtr = Formatter()
    return fmtr.vformat(file_name, (), model_options)


def get_model_options_from_args(args, i_iter=None):
    if isinstance(args, dict) or isinstance(args, defaultdict):
        args_dict = args
    else:
        try:
            args_dict = args.__dict__
        except:
            raise AttributeError('Invalid args!')

    options = defaultdict(lambda: 'None')
    for k, v in args_dict.items():
        if v in [True, 'TRUE', 'True', 'true']:
            if type(v) is bool or type(v) is str:
                options[k] = k
            else:
                options[k] = v
        elif v in [False, 'FALSE', 'False', 'false']:
            if type(v) is bool or type(v) is str:
                continue
            else:
                options[k] = v
        else:
            options[k] = v

    if i_iter is not None:
        options.update({'i_iter': i_iter})
    return options


def load_latest_checkpoints(load_dir, args, logger=logging):
    pattern = r'^((?!best_)(?!optimizer_)(?!centroids_))(?P<model_name>.+)\+(?P<jitter>.+)\+i(?P<in_features>.+)_' + \
              r'(?P<source_dataset>.+)2(?P<target_dataset>.+)_(?P<i_iter>[\d]{6})\.pth$'

    checkpoint_list = []
    for file in os.listdir(load_dir):
        if re.search(pattern, file):
            checkpoint_list.append(file)

    if checkpoint_list:
        checkpoint_list = sorted(checkpoint_list, reverse=True)
        latest_checkpoint = checkpoint_list[0]
        loaded_model = torch.load(os.path.join(load_dir, latest_checkpoint))

        latest_optimizer = 'optimizer_' + checkpoint_list[0]
        try:
            loaded_optimizer = torch.load(os.path.join(load_dir, latest_optimizer))
        except FileNotFoundError:
            logger.warning('best optimizer is not found. Set to default Optimizer states!')
            loaded_optimizer = {}

        if args.sm_loss:
            latest_centroids = 'centroids_' + checkpoint_list[0]
            try:
                loaded_centorids = torch.load(os.path.join(load_dir, latest_centroids))
            except FileNotFoundError:
                logger.warning('best centroids is not found. Set to default centroids!')
                loaded_centorids = {}
        else:
            loaded_centorids = {}
        return loaded_model, loaded_optimizer, loaded_centorids
    else:
        raise FileNotFoundError('latest checkpoints are not found!')


def load_best_checkpoints(load_dir, args, logger=logging):
    pattern = r'^best_((?!optimizer_)(?!centroids_))(?P<model_name>.+)\+(?P<jitter>.+)\+i(?P<in_features>.+)_' + \
              r'(?P<source_dataset>.+)2(?P<target_dataset>.+)\.pth$'
    best_model_ckpt_filename = None
    for file in os.listdir(load_dir):
        if re.search(pattern, file):
            best_model_ckpt_filename = file

    if best_model_ckpt_filename:
        loaded_model = torch.load(os.path.join(load_dir, best_model_ckpt_filename))

        best_optimizer = 'best_optimizer_' + best_model_ckpt_filename[5:]
        try:
            loaded_optimizer = torch.load(os.path.join(load_dir, best_optimizer))
        except FileNotFoundError:
            logger.warning('best optimizer is not found. Set to default Optimizer states!')
            loaded_optimizer = {}

        if args.sm_loss:
            best_centroids = 'best_centroids_' + best_model_ckpt_filename[5:]
            try:
                loaded_centorids = torch.load(os.path.join(load_dir, best_centroids))
            except FileNotFoundError:
                logger.warning('best centroids is not found. Set to default centroids!')
                loaded_centorids = {}
        else:
            loaded_centorids = {}
        return loaded_model, loaded_optimizer, loaded_centorids
    else:
        raise FileNotFoundError('best checkpoints are not found!')


def save_checkpoints(save_dir, options, i_iter, model_dict, optimizer_dict, centroids_dict, logger=logging, best=False):
    save_path = os.path.join(save_dir, get_model_filename(options, best=best))
    logger.info('save {}model: {}'.format('best ' if best else '', save_path))
    model_dict.update({'iteration': i_iter})
    torch.save(model_dict, save_path)

    if optimizer_dict:
        save_path = os.path.join(save_dir, get_optimizer_filename(options, best=best))
        logger.info('save {}optimizer: {}'.format('best ' if best else '', save_path))
        optimizer_dict.update({'iteration': i_iter})
        torch.save(optimizer_dict, save_path)
    if centroids_dict:
        # save centroids
        save_path = os.path.join(save_dir, get_centroids_filename(options, best=best))
        logger.info('save {}centroids: {}'.format('best ' if best else '', save_path))
        centroids_dict.update({'iteration': i_iter})
        torch.save(centroids_dict, save_path)
