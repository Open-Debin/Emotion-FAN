import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
from basic_code import load, util, networks
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    parser = argparse.ArgumentParser(description='PyTorch Frame Attention Network Training')
    parser.add_argument('--epochs', default=180, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=4e-6, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()
    best_acc = 0
    logger = util.Logger('./log/','baseline_afew')
    
    ''' Load data '''
    root_train = './data/face/train_afew'
    list_train = './data/txt/afew_train.txt'
    batchsize_train= 48
    root_eval = './data/face/val_afew'
    list_eval = './data/txt/afew_eval.txt'
    batchsize_eval= 64

    train_loader, val_loader = load.afew_faces_baseline(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval)

    ''' Load model '''
    _structure = models.resnet18(num_classes=7)
    _parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
    model = load.model_parameters(_structure, _parameterDir)

    ''' Loss & Optimizer '''
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    cudnn.benchmark = True

    ''' Train & Eval '''
    if args.evaluate == True:
        logger.print('args.evaluate: {:}', args.evaluate)
        val(val_loader, model, logger)
        return
    logger.print('baseline afew dataset, learning rate: {:}'.format(args.lr))

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, epoch, logger)
        acc_epoch = val(val_loader, model, logger)
        is_best = acc_epoch > best_acc
        if is_best:
            logger.print('better model!')
            best_acc = max(acc_epoch, best_acc)
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'accuracy': acc_epoch,
            }, at_type='baseline')
            
        lr_scheduler.step()
        logger.print("epoch: {:} learning rate:{:}".format(epoch+1, optimizer.param_groups[0]['lr']))
            
def train(train_loader, model, optimizer, epoch, logger):
    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideoSoft = util.AverageMeter()

    # switch to train mode
    output_store_soft = []
    target_store = []
    index_vector = []

    model.train()
    
    for i, (input_var, target_var, index) in enumerate(train_loader):

        target_var = target_var.to(DEVICE)
        input_var = input_var.to(DEVICE)

        # model
        pred_score = model(input_var)
        loss = F.cross_entropy(pred_score, target_var).sum()
        
        output_store_soft.append(F.softmax(pred_score, dim=1))
        target_store.append(target_var)
        index_vector.append(index)
        
        # measure accuracy and record loss
        acc_iter = util.accuracy(pred_score.data, target_var, topk=(1,))
        losses.update(loss.item(), input_var.size(0))
        topframe.update(acc_iter[0], input_var.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            logger.print('Epoch: [{:3d}][{:3d}/{:3d}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_Iter@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, topframe=topframe))

    index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)

    index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
    output_store_soft = torch.cat(output_store_soft, dim=0)
    target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
    output_store_soft = index_matrix.mm(output_store_soft)
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]
    prec_video_soft = util.accuracy(output_store_soft, target_vector, topk=(1,))
    topVideoSoft.update(prec_video_soft[0].item(), i + 1)
    logger.print(' *Acc@Video_soft {topsoft.avg:.3f}   *Acc@Frame {topframe.avg:.3f} '.format(topsoft=topVideoSoft, topframe=topframe))

def val(train_loader, model, logger):
    topframe = util.AverageMeter()
    topVideoSoft = util.AverageMeter()

    # switch to train mode
    output_store_soft = []
    target_store = []
    index_vector = []

    model.eval()
    with torch.no_grad():
        for i, (input_var, target_var, index) in enumerate(train_loader):

            target_var = target_var.to(DEVICE)
            input_var = input_var.to(DEVICE)

            # model
            pred_score = model(input_var)

            output_store_soft.append(F.softmax(pred_score, dim=1))
            target_store.append(target_var)
            index_vector.append(index)

            # measure accuracy and record loss
            acc_iter = util.accuracy(pred_score.data, target_var, topk=(1,))
            topframe.update(acc_iter[0], input_var.size(0))

        index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)

        index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
        output_store_soft = torch.cat(output_store_soft, dim=0)
        target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
        output_store_soft = index_matrix.mm(output_store_soft)
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]
        prec_video_soft = util.accuracy(output_store_soft, target_vector, topk=(1,))
        topVideoSoft.update(prec_video_soft[0].item(), i + 1)
        logger.print(' *Acc@Video {topVideo.avg:.3f} '.format(topVideo=topVideoSoft))

    return topVideoSoft.avg

if __name__ == '__main__':
    main()
