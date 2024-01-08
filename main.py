import yaml
import time
import argparse
import numpy as np
import torch
from dataloader import *
from metrics import test_metrics
from utils import init_seed, load_graph
from trainer import TrainModel


def model_running(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    # load dataset
    dataloader, ytest = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )
    scaler = dataloader['scaler']

    # load graph (normalized)
    adj_mx = load_graph(args.graph_file, device=args.device)
    adj_mx = torch.tensor(adj_mx).to(args.device)
    num_nodes = len(adj_mx)

    engine = TrainModel(args, scaler, adj_mx, num_nodes)
    print("start training...", flush=True)

    his_loss = []
    his_train_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        print('********** Epoch: %03d Start： **********' % i)

        # train stage
        print('***Training Stage:')
        train_loss = []
        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader['train']):
            train_x = torch.Tensor(x).cuda()  # [64, 19, 128, 2]
            train_x = train_x.transpose(1, 3)  # [64, 2, 128, 19]
            train_y = torch.Tensor(y).cuda()  # [64, 1, 128, 2]
            train_y = train_y.transpose(1, 3)  # [64, 2, 128, 1]
            metrics = engine.train(train_x, train_y)
            train_loss.append(metrics)

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(iter, train_loss[-1], flush=True))

        mean_train_loss = np.mean(train_loss)
        his_train_loss.append(mean_train_loss)

        t2 = time.time()
        train_time.append(t2 - t1)

        log = 'Epoch: {:03d}, Mean Train Loss: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mean_train_loss, (t2 - t1)), flush=True)

        # validation stage
        print('***Validation Stage:')
        valid_loss = []
        v1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val']):
            test_x = torch.Tensor(x).cuda()
            test_x = test_x.transpose(1, 3)
            test_y = torch.Tensor(y).cuda()
            test_y = test_y.transpose(1, 3)
            metrics = engine.eval(test_x, test_y)
            valid_loss.append(metrics)

        mean_valid_loss = np.mean(valid_loss)
        his_loss.append(mean_valid_loss)

        v2 = time.time()
        val_time.append(v2 - v1)

        log = 'Epoch: {:03d}, Mean Valid Loss: {:.4f}, Valid Time: {:.4f}/epoch'
        print(log.format(i, mean_valid_loss, (v2 - v1)), flush=True)

        print('********** Epoch: %03d END **********' % i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    outputs = []
    realy = torch.Tensor(ytest).transpose(1, 3).cuda()
    for iter, (x, y) in enumerate(dataloader['test']):
        test_x = torch.Tensor(x).transpose(1, 3).cuda()
        with torch.no_grad():
            preds, _, _ = engine.model(test_x)
            preds = preds.transpose(1, 3)  # [64, 2, 128, 1]
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print("Training finished")

    y_pred = scaler.inverse_transform(yhat).permute(0, 3, 2, 1)
    y_true = scaler.inverse_transform(realy).permute(0, 3, 2, 1)

    print('\n')
    mae_in, mape_in = test_metrics(y_pred[..., 0], y_true[..., 0])
    print("INFLOW:")
    print("MAE: {:.2f}, MAPE: {:.2f}%".format(mae_in, mape_in * 100))

    mae_out, mape_out = test_metrics(y_pred[..., 1], y_true[..., 1])
    print("OUTFLOW:")
    print("MAE: {:.2f}, MAPE: {:.2f}%".format(mae_out, mape_out * 100))


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(open(args.config_filename), Loader=yaml.FullLoader)
    args = argparse.Namespace(**configs)

    model_running(args)

    end_time = time.time()
    print("Total Runtime:{}".format(end_time - start_time))
