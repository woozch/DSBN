"""
Factory method for easily getting discriminator by name.
written by wgchang
"""
from discriminator.discriminator import DigitDiscriminator, OfficeDiscriminator, CPUADiscriminator

__sets = {}

for exp_setting in ['digits', 'office', 'imageclef', 'visda', 'cpua', 'office-home', 'office-caltech']:
    if exp_setting == 'digits':
        name = exp_setting
        __sets[name] = (lambda in_features: DigitDiscriminator(in_features=in_features))
    elif exp_setting in ['office', 'office-caltech']:
        name = exp_setting
        __sets[name] = (lambda in_features: OfficeDiscriminator(in_features=in_features))
    elif exp_setting == 'cpua':
        name = exp_setting
        __sets[name] = (lambda in_features: CPUADiscriminator(in_features=in_features))
    else:
        name = exp_setting
        __sets[name] = (lambda in_features: OfficeDiscriminator(in_features=in_features))


def get_discriminator(exp_setting, in_features):
    if exp_setting not in __sets:
        raise KeyError(
            'Unknown Discriminator: {}, in_features: {}'.format(exp_setting, in_features))
    return __sets[exp_setting](in_features=in_features)
