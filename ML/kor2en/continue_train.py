import sys
import os.path
import torch

from train import define_argparser
from train import main

def overwrite_config(config,prev_config):
    # This method provides a compatibility for new or missing arguments.
    for prev_key in vars(prev_config).keys():
        if not prev_key in vars(config).keys():
            print('WARNING!!! Argument "--%s" is not found in current argument parser.\tIgnore saved value:' % prev_key,
                  vars(prev_config)[prev_key])

    for key in vars(config).keys():
        if not key in vars(prev_config).keys():
            print('WARNING!!! Argument "--%s" is not found in saved model.\tUse current value:' % key,
                  vars(config)[key])

        elif vars(config)[key] != vars(prev_config)[key]:
            if '--%s' % key in sys.argv:
                print('WARNING!!! You changed value for argument "--%s".\tUse current value:' % key,
                      vars(config)[key])

            else:
                vars(config)[key] = vars(prev_config)[key]

    return config

def continue_main(config,main):
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(config.load_fn, map_location='cpu')

        prev_config = saved_data['config']
        config = overwrite_config(config,prev_config)

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        main(config,model_weight-model_weight,opt_weight=opt_weight)

    else:
        print('Cannot find file %s' %config.load_fn)


if __name__ == '__main__':
    config = define_argparser(is_continue=True)
    continue_main(config,main)