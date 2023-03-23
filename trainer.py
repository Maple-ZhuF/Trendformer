import torch.optim as optim
import math
from model import *
import utils

class Trainer():
    def __init__(self, model, lrate, wdecay, clip,  scaler, device):
        self.scaler = scaler
        # self.model = model
        self.model = nn.DataParallel(model, device_ids=[0,1])
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = utils.masked_mae
        self.clip = clip

    def train(self, input, trend, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, trend)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)   #torch.Size([64, 1, 207, 12])
        # print(real.size())
        # print(predict.size())

        loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = utils.masked_mape(predict,real,0.0).item()
        rmse = utils.masked_rmse(predict,real,0.0).item()

        return loss.item(),mape,rmse

    def eval(self, input, trend, real_val):
        self.model.eval()
        output = self.model(input,trend)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = utils.masked_mape(predict,real,0.0).item()
        rmse = utils.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
