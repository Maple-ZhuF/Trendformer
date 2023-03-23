import argparse
import math
import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import SparseGraphTransformer
import numpy as np


import utils
import os
from torch.nn import init


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='../data/pems08/',help='data path')
parser.add_argument('--seq_len',type=int,default=12,help='')
parser.add_argument('--out_len',type=int,default=12,help='')
parser.add_argument('--nid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=10,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./garage/pems04/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

#set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def eval_epoch(model, dataloader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    scaler = dataloader['scaler']
    compute_loss = utils.masked_mae

    # validation
    valid_mae = []
    valid_mape = []
    valid_rmse = []

    for iter, (x, y,trend) in enumerate(tqdm(dataloader['val_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        testy = testy[:, 0, :, :]
        test_trend = torch.Tensor(trend).to(device)
        test_trend = test_trend.transpose(1,2)

        with torch.no_grad():
            output = model(testx, test_trend)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        predict = scaler.inverse_transform(output)

        real = torch.unsqueeze(testy, dim=1) #(num_samples, 1, num_nodes, input_length)
        loss = compute_loss(predict, real, 0.0)

        mae = loss.item()
        mape = utils.masked_mape(predict,real,0.0).item()
        rmse = utils.masked_rmse(predict,real,0.0).item()

        valid_mae.append(mae)
        valid_mape.append(mape)
        valid_rmse.append(rmse)

    return valid_mae, valid_mape, valid_rmse

def eval_epoch2(model, dataloader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    scaler = dataloader['scaler']
    compute_loss = utils.masked_mae

    print("validation...", flush=True)

    outputs = []
    real_y = torch.Tensor(dataloader['y_val']).to(device)
    real = real_y.transpose(1,3)[:,0,:,:] # (num_samples, input_dim, num_nodes, input_length) --> (num_samples, num_nodes, input_length)

    for iter, (x, y,trend) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        test_trend = torch.Tensor(trend).to(device)
        test_trend = test_trend.transpose(1,2)
        with torch.no_grad():
            preds = model(testx,test_trend) #output = [batch_size,1,num_nodes,12]
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    pred = yhat[:real_y.size(0), ...]
    pred = scaler.inverse_transform(pred)
    # pred = yhat

    loss = compute_loss(pred, real, 0.0)
    mae = loss.item()
    mape = utils.masked_mape(pred,real,0.0).item()
    rmse = utils.masked_rmse(pred,real,0.0).item()

    # mae, mape, rmse = utils.metric(pred, real)


    return mae, mape, rmse


def train_epoch(model, optimizer, dataloader, args, device):
    ''' Epoch operation in training phase'''

    model.train()
    clip = 5
    scaler = dataloader['scaler']
    compute_loss  = utils.masked_mae

    train_mae = []
    train_mape = []
    train_rmse = []

    dataloader['train_loader'].shuffle()
    for iter, (x, y, trend) in enumerate(dataloader['train_loader'].get_iterator()):

        # prepare data
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        trainy = trainy[:, 0, :, :]
        train_trend = torch.Tensor(trend).to(device)
        train_trend = train_trend.transpose(1,2)

        # x: (num_samples, input_length, num_nodes, input_dim) --> (num_samples, input_dim, num_nodes, input_length)
        # y: (num_samples, output_length, num_nodes, output_dim) --> (num_samples, output_dim, num_nodes, output_length) --> (num_samples, num_nodes, output_length)
        # trend: (num_samples, output_length, num_nodes) --> (num_samples, num_nodes, output_length,)
        # print('y_example',trainy[0,0,:])
        # print('y_trend',train_trend[0,0,:])

        #foward
        optimizer.zero_grad()
        output = model(trainx, train_trend) # history and future trend
        output = output.squeeze()
        # (num_samples,  num_nodes, out_len)

        predict = scaler.inverse_transform(output)


        #real = torch.unsqueeze(trainy, dim=1) #(num_samples, 1, num_nodes, input_length)
        # print('out_size',output.size())
        # print('real_size',real.size())
        real = trainy
        loss = compute_loss(predict, real, 0.0)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        mae = loss.item()
        mape = utils.masked_mape(predict,real,0.0).item()
        rmse = utils.masked_rmse(predict,real,0.0).item()

        train_mae.append(mae)
        train_mape.append(mape)
        train_rmse.append(rmse)


        if iter % args.print_every == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_mae[-1], train_mape[-1], train_rmse[-1]), flush=True)


    return train_mae, train_mape, train_rmse



def train(model, optimizer,dataloader, args, device):

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    print(model)

    ''' Start training '''

    print("Start training...", flush=True)


    history_loss = []
    train_time = []
    for epoch_i in range(1, args.epochs + 1):
        print('[ Epoch', epoch_i, ']')

        start_time = time.time()

        train_mae, train_mape, train_rmse  = train_epoch(model, optimizer, dataloader, args, device)
        s1 = time.time()
        valid_mae, valid_mape, valid_rmse = eval_epoch2(model,  dataloader, device)

        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)


        end_time = time.time()

        log = 'Epoch: {:03d}/{}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch_i,args.epochs, mtrain_mae, mtrain_mape, mtrain_rmse, (end_time - start_time)),
              flush=True)
        log = 'Epoch: {:03d}/{}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Inference Time: {:.4f}/epoch'
        print(log.format(epoch_i,args.epochs, valid_mae, valid_mape, valid_rmse, (end_time - s1)),
              flush=True)

        save_path = args.save + "batch_size{}/".format(args.batch_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(model.state_dict(), save_path +"epoch_"+str(epoch_i)+"_"+str(round(valid_mae,2))+".pth")

        history_loss.append(valid_mae)
        train_time.append(end_time-start_time)
        curr_best_epoch = np.argmin(history_loss) +1
        if((epoch_i - curr_best_epoch)>20):        #如果10个epochs后valid loss没有下降，则停止训练
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    if(epoch_i < args.epochs):
        print('early stop!')
    print("Training finished")
    best_epoch = np.argmin(history_loss) + 1
    print("The valid loss on best epoch{} is {}".format(best_epoch,str(round(history_loss[best_epoch-1], 4))))

    return history_loss


def test(model, dataloader, args, device, history_loss):

    ''' testing '''
    model.eval()

    scaler = dataloader['scaler']

    print("testing...", flush=True)

    best_epoch = np.argmin(history_loss)+1
    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    save_path = args.save + "batch_size{}/".format(args.batch_size)
    model.load_state_dict(torch.load(save_path +"epoch_"+str(best_epoch)+"_"+str(round(history_loss[best_epoch-1],2))+".pth"))
    # model.load_state_dict(torch.load(save_path +"epoch_"+str(best_epoch)+"_"+str(round(3.07,2))+".pth"))
    outputs = []
    real_y = torch.Tensor(dataloader['y_test']).to(device)
    real_y = real_y.transpose(1,3)[:,0,:,:] # (num_samples, input_dim, num_nodes, input_length) --> (num_samples, num_nodes, input_length)
    # trend_y = torch.Tensor(dataloader['ytrend_test']).to(device)  # (num_samples, input_length num_nodes,)
    # trend_y = trend_y.transpose(1,2) # (num_samples, num_nodes, input_length)

    for iter, (x, y,trend) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        test_trend = torch.Tensor(trend).to(device)
        test_trend = test_trend.transpose(1,2)
        with torch.no_grad():
            preds = model(testx,test_trend) #output = [batch_size,1,num_nodes,12]
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:real_y.size(0), ...]
    print(yhat.size())
    print(real_y.size())

    for i in range(real_y.size(2)):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = real_y[:, :, i]
        metrics = utils.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
        print(log.format(i+1 , metrics[0], metrics[1]*100, metrics[2]))

    torch.save(model.state_dict(),
               save_path + "exp" + str(args.expid) + "_best_" + str(round(history_loss[best_epoch-1], 2)) + ".pth")

    print(' ')
    if(args.out_len>10):
        for i in [2,5,11]:
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = real_y[:, :, i]
            metrics = utils.metric(pred, real)
            log = 'Evaluate for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
            print(log.format(i+1 , metrics[0], metrics[1]*100, metrics[2]))

    pred = scaler.inverse_transform(yhat)
    metrics = utils.metric(pred, real_y)
    log = 'Average:, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
    print(log.format(metrics[0], metrics[1]*100, metrics[2]))



# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    # elif isinstance(m, nn.Linear):
    #     m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()

if __name__ == "__main__":
    t1 = time.time()

    """
    dataset          traffic     solar       electricity     covid19   pems03    pems04    pems07   pems08
    granularity      1hour       10minutes     1hour          1day      1hour     1hour    1hour    1hour
    samples          17554        52560        26304          298        358       307     883       170
    nodes             862          137          321           191       26208     16992    28224    17856
    """

    device_id = 3
    model_parallel = False
    args.batch_size = 32
    d_model = 16
    n_layers = 2
    args.epochs = 100
    args.learning_rate = 0.001
    args.num_nodes = 170
    range_size = 10
    dataset = 'pems08'
    if_ffn = True
    loss_func = 'mae'

    print('batch_size {} , d_model {} , learing_rate {} , if_ffn {}, n_layers {}'.format(args.batch_size,d_model,args.learning_rate,if_ffn,n_layers))
    print('dataset {}, num_nodes {}, range_size {} loss {}'.format(dataset, args.num_nodes, range_size, loss_func))

    args.data = '../data/{}/'.format(dataset)
    args.save = './garage/{}/'.format(dataset)
    args.device = 'cuda:{}'.format(device_id)
    device = torch.device(args.device)


    dataloader = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)

    #d_inner为feed forward中变换的维度
    model = SparseGraphTransformer(
        in_dim=2, out_dim=1, seq_len=12, out_len=12, end_channels = 64,d_inner=32,
        en_layers=n_layers,  de_layers=n_layers, d_model=d_model,
        num_nodes=args.num_nodes, range_size=range_size, dropout=0.3,if_ffn=if_ffn
    )
    print(args)
    if model_parallel:
        model = nn.DataParallel(model,device_ids=[device_id,device_id+1])
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr= args.learning_rate, weight_decay=args.weight_decay)

    # optimizer = ScheduledOptim(
    #     optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     2.0, opt.d_model, opt.n_warmup_steps)

    history_loss = train(model, optimizer, dataloader, args, device)
    test(model, dataloader, args, device, history_loss)

    t2 = time.time()
    print("Total time spent: {:.4f} minutes".format((t2-t1)/60))














