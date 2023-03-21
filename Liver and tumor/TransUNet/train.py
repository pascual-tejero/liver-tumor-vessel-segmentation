import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


#----------
# Change to run in Polyaxon
#----------

#----------
# Libraries for importing files
#----------
import sys
import os
import yaml
from yaml.loader import SafeLoader

#----------
# Local run
#----------

# sys.path.append('C:\\Users\\Natalia\\Documents\\01. Master\\2022-WS\\02. CS\\06. Code\\model')
# sys.path.append('C:\\Users\\Natalia\\Documents\\01. Master\\2022-WS\\02. CS\\06. Code\\data')
CONFIG_PATH = 'C:\\Users\\Natalia\\Documents\\01. Master\\2022-WS\\02. CS\\06. Code\\TransUNet-main\\TransUNet\\config'


#-------------
# Polyaxon run 
#-------------

# sys.path.append('./model')
# sys.path.append('./data')
# sys.path.append('./src/metrics')
CONFIG_PATH = './config'

from polyaxon_client.tracking import Experiment

if __name__ == "__main__":


    with open(os.path.join(CONFIG_PATH, 'config_dncnn.yaml')) as f:
        hparams = yaml.load(f, Loader=SafeLoader)
    
    if hparams['on_polyaxon']:
        
        experiment = Experiment()
        data_paths = experiment.get_data_paths()
        # root_files = data_paths['data1']+ hparams['polyaxon_dataset']
        pretrained_model = data_paths['data1']+ hparams['polyaxon_pretrained_model']

        gpus = hparams["gpus_polyaxon"]
        num_workers = hparams["num_workers_polyaxon"]

        if args.dataset == 'Synapse':
            root_files = data_paths['data1']+ hparams['polyaxon_synapse_dataset']
        elif args.dataset == 'LITS':
            root_files = data_paths['data1']+ hparams['polyaxon_lits_dataset']

    else:
        # root_files = hparams['local_dataset']
        gpus = hparams["gpus_local"]
        num_workers = hparams["num_workers_local"]
        pretrained_model = hparams['pretrained_model']

        if args.dataset == 'Synapse':
            root_files = hparams['synapse_dataset']
        elif args.dataset == 'LITS':
            root_files = hparams['lits_dataset']


#----------
# End Change to run in Polyaxon
#----------

print(root_files)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # change adding
    cuda = torch.device('cuda')
    #end change
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': root_files+'/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        #CHANGE LITS
        'LITS': {
            'root_path': root_files+'/train_npz',
            'list_dir': './lists/lists_LITS',
            'num_classes': 3,
        },
        #END CHANGE
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    ##Change
    today = datetime.now().strftime('%Y%m%d %H:%M:%S')
    today = np.str_(np.char.replace(np.char.replace(today,' ','_'),':',"_"))
    print(today)
    #end change
    # snapshot_path = snapshot_path + '_' + today
    
    snapshot_path = "../{}/{}/{}".format(pretrained_model, args.exp,today+ '_TU')
    print(snapshot_path)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    print(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda() 
    # CHANGE
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(cuda)

    net.load_from(weights=np.load(pretrained_model + config_vit.pretrained_path))
    
    trainer = {'Synapse': trainer_synapse,
                #CHANGE LITS 
                'LITS': trainer_synapse,
                #END CHANGE
                }
    trainer[dataset_name](args, net, snapshot_path,num_workers) #CHANGE ADD NUM_WORKERS AS PARAMETER