import torch
import torch.optim as optim
from metrics import masked_mae_loss
from model import UrbanGCL


class TrainModel:
    def __init__(self, args, scaler, adj_mx, num_nodes):
        self.model = UrbanGCL(args, adj_mx, num_nodes).to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.lr_decay)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.scaler = scaler
        self.yita = args.yita
        self.lamda1 = args.lamda1
        self.lamda2 = args.lamda2
        self.clip = args.clip

    def train(self, input_val, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        # input_val:[64, 2, 128, 19] -> pre_val:[64, 1, 128, 2]
        predict, sc_loss, tc_loss = self.model(input_val)
        predict = predict.transpose(1, 3)  # [64, 2, 128, 1]

        pre_val = self.scaler.inverse_transform(predict).permute(0, 3, 2, 1)
        real_val = self.scaler.inverse_transform(real_val).permute(0, 3, 2, 1)

        loss = self.yita * self.mae(pre_val[..., 0], real_val[..., 0]) + \
               (1 - self.yita) * self.mae(pre_val[..., 1], real_val[..., 1])

        (loss + self.lamda1 * sc_loss + self.lamda2 * tc_loss).backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        return loss.item()

    def eval(self, input_val, real_val):
        self.model.eval()

        predict, _, _ = self.model(input_val)
        predict = predict.transpose(1, 3)

        pre_val = self.scaler.inverse_transform(predict).permute(0, 3, 2, 1)
        real_val = self.scaler.inverse_transform(real_val).permute(0, 3, 2, 1)

        loss = self.yita * self.mae(pre_val[..., 0], real_val[..., 0]) + \
               (1 - self.yita) * self.mae(pre_val[..., 1], real_val[..., 1])
        return loss.item()