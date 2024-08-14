'''
Description: pre-train model with RSI Dataset
'''
import torch
import os
import rsiattack
from rsiattack.model import func
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F
import csv
import uuid
import datetime


model_box = func.models_mapping

parser = argparse.ArgumentParser(description="training the models to attack")
parser.add_argument('--data_type', type=str, default='MTARSI',
                    help='used trainset in training')
parser.add_argument('--data_dir', type=str, default='./dataset',
                    help='path of the dataset')
parser.add_argument('--log_path', type=str, default='logs/train_logs.csv')
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='trained model where to save')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=18)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
if args.data_type == 'FGSCR_42':
    args.num_classes = 42
elif args.data_type == 'MTARSI': 
    args.num_classes = 20

def train():
    """ train backbone models """
    train_set = rsiattack.trainDataset('train', args)
    test_set = rsiattack.trainDataset('test', args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=5)
    test_loader = DataLoader(test_set,batch_size=args.batch_size,
                              shuffle=False, num_workers=5)
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    for key, fc in model_box.items():
        net = fc(args.num_classes) # get net Class from model_box
        optimizer = torch.optim.SGD(net.parameters(),lr=args.lr)
        scheduler = StepLR(optimizer,step_size=30,gamma=0.1)
        loss = torch.nn.CrossEntropyLoss()
        net = net.to(args.device)
        print('** start training {}, Data is {} **'.format(key, args.data_type))
        for _ in tqdm(range(args.epochs)):
            net.train()
            for data, gt in train_loader:
                data, gt = data.to(args.device), gt.to(args.device)
                y_hat = net(data)
                pred = torch.argmax(y_hat,1)
                lossrt = loss(y_hat, gt)
                optimizer.zero_grad()
                lossrt.backward()
                optimizer.step()

            scheduler.step()
        
        """ eval model and print eval result """
        net.eval()
        with torch.no_grad():
            correct_pred = 0
            for data, target in train_loader:
                data, target = data.to(args.device), target.to(args.device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                correct_pred += (pred == target).sum()
            train_acc = correct_pred.item() / len(train_set)

        with torch.no_grad():
            correct_pred = 0
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                correct_pred += (pred == target).sum()
            test_acc = correct_pred.item() / len(test_set)
        
        if not os.path.exists(args.log_path):
            with open(args.log_path, 'w') as f:
                csv_writer = csv.writer(f)
                log_head = ['model_name', 'model_type', 'train_type', 'create_time',
                             'train_acc(%)', 'test_acc(%)', 'save_path', 'backbone']
                csv_writer.writerow(log_head)
        with open(args.log_path, 'a+') as f:
            # write logs in log_path.csv
            random_id = str(uuid.uuid4())
            dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = key + '_' + args.data_type + '.pt'
            save_path = os.path.join(args.save_dir, file_name)
            row_log = [file_name, key, args.data_type, dtime,
                       '{:.4f}'.format(train_acc * 100), 
                       '{:.4f}'.format(test_acc * 100),
                       save_path, str(args.backbone)]
            
            csv_writer = csv.writer(f)
            csv_writer.writerow(row_log)

            torch.save(net.state_dict(), save_path)
        print('training result is trainacc = {:04f}% , \
              testacc = {:04f}%'.format(train_acc* 100, test_acc*100))
    print('__finish training__')

if __name__ == '__main__':
    train()