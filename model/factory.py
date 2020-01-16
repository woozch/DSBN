"""
Factory method for easily getting model by name.
written by wgchang
"""
from model import lenet, alexnet, resnet, resnetdsbn
__sets = {}

for model_name in ['lenet', 'lenetdsbn', 'alexnet', 'resnet50', 'resnet101', 'resnet152', 'resnet50dsbn',
                   'resnet101dsbn',  'resnet152dsbn']:

    if model_name == 'lenet':
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains:
                              lenet.LeNet(num_classes=num_classes, in_features=in_features,
                                          weights_init_path=None))
    elif model_name == 'lenetdsbn':
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains:
                              lenet.DSBNLeNet(num_classes=num_classes, in_features=in_features,
                                              weights_init_path=None, num_domains=num_domains))

    elif model_name == 'alexnet':
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains:
                              alexnet.AlexNet(num_classes=num_classes, in_features=in_features,
                                              weights_init_path='data/pretrained/bvlc_alexnet_pytorch.pth')
                              if pretrained else
                              alexnet.AlexNet(num_classes=num_classes, in_features=in_features,
                                              weights_init_path=None))
    elif model_name in ['cpuanet50', 'cpuanet101', 'cpuanet152']:
        eval_str = "cpuanet.{}"
        __sets[model_name] = (
            lambda num_classes, in_features, pretrained, num_domains, model_name=model_name, eval_str=eval_str:
            eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                              in_features=in_features))

    elif model_name in ['cpuanet50dsbn', 'cpuanet101dsbn', 'cpuanet152dsbn']:
        eval_str = "cpuanetdsbn.{}"
        __sets[model_name] = (
            lambda num_classes, in_features, pretrained, num_domains, model_name=model_name, eval_str=eval_str:
            eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                              in_features=in_features, num_domains=num_domains))
    elif model_name in ['resnet50', 'resnet101', 'resnet152']:
        eval_str = "resnet.{}"
        __sets[model_name] = (
            lambda num_classes, in_features, pretrained, num_domains, model_name=model_name, eval_str=eval_str:
            eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                              in_features=in_features))

    elif model_name in ['resnet50dsbn', 'resnet101dsbn', 'resnet152dsbn']:
        eval_str = "resnetdsbn.{}"
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains,
                                     model_name=model_name, eval_str=eval_str:
                              eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                                                in_features=in_features, num_domains=num_domains))

    elif model_name in ['resnet50dsbn-multi', 'resnet101dsbn-multi', 'resnet152dsbn-multi']:
        eval_str = "resnetdsbn.{}"
        __sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains,
                                     model_name=model_name, eval_str=eval_str:
                              eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                                                in_features=in_features, num_domains=num_domains))



def get_model(model_name, num_classes, in_features=0, num_domains=2, pretrained=False):
    model_key = model_name
    if model_key not in __sets:
        raise KeyError(
            'Unknown Model: {}, num_classes: {}, in_features: {}'.format(model_key, num_classes, in_features))
    return __sets[model_key](num_classes=num_classes, in_features=in_features,
                             pretrained=pretrained, num_domains=num_domains)
