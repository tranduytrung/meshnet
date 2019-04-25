import copy
import os, glob
from datetime import datetime
import torch, torch.jit
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map
import numpy as np

def trace(model, batch_size):
    # trace
    centers = torch.rand((batch_size, 3, 1024), dtype=torch.float)
    corners = torch.rand((batch_size, 9, 1024), dtype=torch.float)
    normals = torch.rand((batch_size, 3, 1024), dtype=torch.float)
    neighbor_index = torch.randint(1024, (batch_size, 1024, 3), dtype=torch.long)
    if torch.cuda.is_available():
        centers = centers.cuda()
        normals = normals.cuda()
        corners = corners.cuda()
        neighbor_index = neighbor_index.cuda()
    
    model.train()
    model_train = torch.jit.trace(model, (centers, corners, normals, neighbor_index))
    
    model.eval()
    model_eval = torch.jit.trace(model, (centers, corners, normals, neighbor_index))
    return model_train, model_eval

def train_model(model, data_loader, criterion, optimizer, scheduler, cfg, start_epoch=1):
    from tensorlog import summary

    best_acc = 0.0
    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    max_epoch = cfg['max_epoch']
    ckpt_root = cfg['ckpt_root']
    batch_size = cfg['batch_size']
    val_acc_hist = []
	
    model_train, model_eval = trace(model, batch_size)

    for epoch in range(start_epoch, max_epoch + 1):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, max_epoch))
        print('-' * 60)

        for phrase in ['train', 'test']:

            if phrase == 'train':
                scheduler.step()
                model.train()
                forward = model_train
            else:
                model.eval()
                forward = model_eval

            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            dataset_size = len(data_loader[phrase].dataset)
            total_steps = int(dataset_size / batch_size)

            for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):
                optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    centers = centers.cuda()
                    normals = normals.cuda()
                    corners = corners.cuda()
                    neighbor_index = neighbor_index.cuda()
                    targets = targets.cuda()

                # centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
                # corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
                # normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
                # neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
                # targets = Variable(torch.cuda.LongTensor(targets.cuda()))

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs, feas = forward(centers, corners, normals, neighbor_index)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phrase == 'test':
                        ft_all = append_feature(ft_all, feas.cpu().numpy())
                        lbl_all = append_feature(lbl_all, targets.cpu().numpy(), flaten=True)

                    batch_loss = loss.item()
                    batch_correct = torch.sum(preds == targets.data).item()
                    batch_acc = batch_correct / batch_size
                    running_loss += batch_loss * batch_size
                    running_corrects += batch_correct

                print(f'{datetime.now()} {phrase} {i}/{total_steps}  loss: {batch_loss:.4f} acc: {batch_acc:.4f}')
                summary.add_scalar(f'{phrase}/batch/loss', batch_loss, global_step=(epoch - 1)*total_steps + i)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            if phrase == 'train':
                print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(datetime.now(), phrase, epoch_loss, epoch_acc))
                print('================================')

            if phrase == 'test':
                epoch_map = calculate_map(ft_all, lbl_all)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    val_acc_hist.append(epoch_acc)
                if epoch_map > best_map:
                    best_map = epoch_map
                if epoch % 10 == 0:
                    filename = os.path.join(ckpt_root, f'{epoch:04d}.pkl')
                    torch.save(copy.deepcopy(model.state_dict()), filename)

                print('{} Loss: {:.4f} Acc: {:.4f} mAP: {:.4f}'.format(phrase, epoch_loss, epoch_acc, epoch_map))
                summary.add_scalar(f'{phrase}/epoch/mAP', epoch_map, global_step=epoch*total_steps)
            
            summary.add_scalar(f'{phrase}/epoch/accuracy', epoch_acc, global_step=epoch*total_steps)
            summary.add_scalar(f'{phrase}/epoch/loss', epoch_loss, global_step=epoch*total_steps)

    summary.close()
    return best_model_wts, val_acc_hist

def load_last(model, ckpt_root):
    pattern = ckpt_root + r'????.pkl'
    ckpts = glob.glob(pattern)
    if len(ckpts) == 0:
        return model
    ckpts.sort()
    last_ckpt = ckpts[-1]
    state_dict = torch.load(last_ckpt, 
        map_location=lambda storage, location: storage.cuda() if torch.cuda.is_available() else storage)
    model.load_state_dict(state_dict)
    print(f'loaded {last_ckpt}')

    # get last epoch
    last_epoch = os.path.basename(last_ckpt).split('.')[0]
    last_epoch = int(last_epoch)

    # get acc hist
    acc_hist_path = os.path.join(ckpt_root, 'acc_hist.npy')
    if os.path.isfile(acc_hist_path):
        acc_hist = np.load(acc_hist_path)
    else:
        acc_hist = []
    return model, last_epoch, acc_hist

def main():
    cfg = get_train_config()

    data_set = {
        x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
    }
    data_loader = {
        x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
        for x in ['train', 'test']
    }

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    # enable cuda if available
    if torch.cuda.is_available():
        model = model.cuda()
    # model = nn.DataParallel(model)
		
    ckpt_root = cfg['ckpt_root']
    model, last_epoch, acc_hist = load_last(model, ckpt_root)
    print(f'last epoch is {last_epoch}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=cfg['gamma'])
    for _ in range(last_epoch):
        scheduler.step()

    best_model_wts, new_acc_hist = train_model(model, data_loader, criterion, optimizer, scheduler, cfg, start_epoch=last_epoch + 1)

    # save acc hist
    acc_hist_path = os.path.join(ckpt_root, 'acc_hist.npy')
    acc_hist = np.concatenate((acc_hist, new_acc_hist))
    np.save(acc_hist_path, acc_hist)

    torch.save(best_model_wts, os.path.join(ckpt_root, 'best_model.pkl'))

if __name__ == '__main__':
    main()