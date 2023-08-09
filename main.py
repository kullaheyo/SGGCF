import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
from itertools import product
import time
import loss
from tensorboardX import SummaryWriter
from utils.utils import check_overfitting, early_stop
from utils import logger
from configure import CONFIG
from dataset import *
from model import *
from test import *
from metric import Recall, NDCG

setproctitle.setproctitle(f"train{CONFIG['name']}")
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
device = torch.device('cuda')
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    print('start training')
    log_interval = CONFIG['log_interval']
    model.train()
    for i, data in enumerate(loader):
        groups_i, items = data
        modelout = model(groups_i.to(device), items.to(device), 'g')
        loss = loss_func(modelout, batch_size=loader.batch_size)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % log_interval == 0:
            print('G_I Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                100. * (i+1) / len(loader), loss))
        
    print('Train Epoch: {}'.format(epoch))
    return loss

group_train_data, group_test_data, user_data, assist_data = get_dataset(CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['task'])
train_loader = DataLoader(group_train_data, 2048, True,
                              num_workers=8, pin_memory=True)
test_loader = DataLoader(group_test_data, 256, False,
                             num_workers=2, pin_memory=True)
u_train_loader = DataLoader(user_data, 2048, True,
                              num_workers=8, pin_memory=True)
gi_graph = group_train_data.ground_truth_g_i
ui_graph = user_data.ground_truth_u_i
gu_graph = assist_data.ground_truth_g_u

metrics = [ NDCG(20), NDCG(50), Recall(20), Recall(50)]
TARGET = 'Recall@50'
loss_func = loss.BPRLoss('mean')
log = logger.Logger(os.path.join(
    CONFIG['log'], CONFIG['dataset_name'], 
    f"{CONFIG['model']}_{CONFIG['task']}", ''), 'best', checkpoint_target=TARGET)

time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

for lr, decay, message_dropout, node_dropout, alpha, cl_reg,  cl_2_reg, cl_temp, cl_2_temp, drop_rate, aug_type\
        in product(CONFIG['lrs'], CONFIG['decays'], CONFIG['message_dropouts'], CONFIG['node_dropouts'], CONFIG['alphas'], CONFIG['cl_regs'], CONFIG['cl_2_regs'] , CONFIG['cl_temps'],  CONFIG['cl_2_temps'], CONFIG['drop_rates'], CONFIG['aug_types']):
    visual_path =  os.path.join(CONFIG['visual'], 
                                CONFIG['dataset_name'],  
                                f"{CONFIG['model']}_{CONFIG['task']}", 
                                f"{time_path}@{CONFIG['note']}", 
                                f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}")

    graph = [gu_graph, ui_graph, gi_graph]
    info = SGGCF_Info(64, decay, message_dropout, node_dropout, alpha, cl_reg, cl_2_reg, cl_temp, cl_2_temp, drop_rate, aug_type, 2)
    model = SGGCF(info, assist_data, graph, device).to(device)
    
    #def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, alpha, cl_reg, ssl_temp, num_layers, act=nn.LeakyReLU()):
    
    
    # op
    op = optim.Adam(model.parameters(), lr=lr)
    # env
    env = {'lr': lr,
           'op': str(op).split(' ')[0],   # Adam
           'dataset': CONFIG['dataset_name'],
           'model': CONFIG['model'], 
           'sample': CONFIG['sample'],
           }

    retry = CONFIG['retry']  # =1
    while retry >= 0:
        # log
        log.update_modelinfo(info, env, metrics)
        try:
            # train & test
            early = CONFIG['early']  
            train_writer = SummaryWriter(log_dir=visual_path, comment='train')
            test_writer = SummaryWriter(log_dir=visual_path, comment='test') 
            print(CONFIG['epochs'])
            for epoch in range(CONFIG['epochs']):
                # train
                print(CONFIG['epochs'])
                trainloss = train(model, epoch+1, train_loader, op, device, CONFIG, loss_func)
                print(trainloss)
                train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)

                # test
                if epoch % CONFIG['test_interval'] == 0:  
                    output_metrics = test(model, test_loader, device, CONFIG, metrics)

                    for metric in output_metrics:
                        test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric}, epoch)
                        if metric==output_metrics[0]:
                            test_writer.add_scalars('metric/single', {metric.get_title(): metric.metric}, epoch)

                    # log
                    log.update_log(metrics, model) 

                    # check overfitting
                    if epoch > 20:
                        if check_overfitting(log.metrics_log, TARGET, 0.001, show=True):
                            break
                    # early stop
                    early = early_stop(
                        log.metrics_log[TARGET], early, threshold=0.001)
                    if early <= 0:
                        break
            train_writer.close()
            test_writer.close()

            log.close_log(TARGET)
            retry = -1
        except RuntimeError:
            retry -= 1