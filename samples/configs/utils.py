import os
import time
import yaml


def convert_config(config):
    config['shuffle'] = bool(config['shuffle'])
    for n in range(1, config['n_layers'] + 1):        
        reg = config['layer_{}_reg'.format(n)]
        extras = {'name' : reg }
        if reg == 'dropout':
            extras['rate'] = config['dropout_rate_{}'.format(n)]
        config['layer_{}_extras'.format(n)] = extras

    return config


def create_yaml_config(run_id, dataset, config, max_epoch):
    
    try:
        with open('samples/configs/{}.yaml'.format(dataset)) as template:
            conf_dict = yaml.load(template, Loader=yaml.FullLoader)
        
        # update configuration
        conf_dict['trainer']['num_epochs'] = max_epoch
        milestones = [int(0.3 * max_epoch), int(0.6 * max_epoch), int(0.8 * max_epoch)]
        conf_dict['scheduler']['milestones'] = milestones

        conf_dict['trainer']['output_dir'] = 'experiments/cifar10-{}'.format(run_id)
        
        if "batch_size" in config:
            batch_size = config['batch_size']
            conf_dict['dataset']['batch_size'] = batch_size

        if "lr" in config:
            lr = config['lr']
            conf_dict['optimizer']['lr'] = lr

        if "weight_decay" in config:
            wd = config['weight_decay']
            conf_dict['optimizer']['weight_decay'] = wd

        if 'optimizer' in config:
            opt = config['optimizer']
            conf_dict['optimizer']['name'] = opt

        if 'model_type' in config:
            m = config['model_type']
            conf_dict['model']['name'] = 'efficientnet_{}'.format(m)

    except Exception as ex:
        warn("Invalid configuration: {}".format(config))
    
    cfg_path = "temp/cifar10-{}.yaml".format(run_id)

    with open(cfg_path, 'w') as f:
        yaml.dump(conf_dict, f)

    return cfg_path

