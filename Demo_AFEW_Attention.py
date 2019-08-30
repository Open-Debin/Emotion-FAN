from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from Code import load_materials, util, Model

#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',
                    help= '0 is self-attention; 1 is relation-attention')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=4e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0

args = parser.parse_args()
'''MyNote '''
at_type = ['self-attention', 'relation-attention'][args.at_type]
print('The attention is ' + at_type)

def main():
    global args, best_prec1

    print('learning rate:', args.lr)
    ''' Load data '''
    arg_rootTrain = './Data/Train'
    arg_listTrain = './Data/list_train.txt'
    arg_batchsize_train= 48

    arg_rooteval = './Data/Val'
    arg_listeval = './Data/list_eval.txt'
    arg_batchsize_eval= 64

    train_loader, val_loader = load_materials.LoadAFEW(arg_rootTrain, arg_listTrain, arg_batchsize_train, arg_rooteval, arg_listeval, arg_batchsize_eval)

    ''' Load model '''
    _structure = Model.resnet18_AT(at_type=at_type)
    _parameterDir = './Parameter/Resnet18_FER+_pytorch.pth.tar'
    model = load_materials.LoadParameter(_structure, _parameterDir)

    ''' Loss & Optimizer '''
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    ''' Train & Eval '''
    print('args.evaluate', args.evaluate)
    if args.evaluate == True:
        validate(val_loader, model)
        return
    print('args.lr', args.lr)

    for epoch in range(args.epochs):
        util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model)

        is_best = prec1 > best_prec1
        if is_best:
            print('better model!')

        best_prec1 = max(prec1, best_prec1)

        if is_best:
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1,
            }, at_type=at_type)
        else:
            print('Model too bad & not save')

def train(train_loader, model, criterion, optimizer, epoch):
    global record_
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []
    index_vector = []

    model.train()
    end = time.time()
    for i, (input_first, target_first, input_second, target_second, input_third, target_third, index) in enumerate(train_loader):

        target = target_first.cuda(async=True)
        input_var = torch.autograd.Variable(torch.stack([input_first, input_second , input_third], dim=4))
        target_var = torch.autograd.Variable(target)
        # compute output
        ''' model & full_model'''
        pred_score = model(input_var)

        loss = criterion(pred_score,target_var)
        loss = loss.sum()
        #
        output_store_fc.append(pred_score)
        target_store.append(target)
        index_vector.append(index)
        # measure accuracy and record loss
        prec1 = util.accuracy(pred_score.data, target, topk=(1,))
        losses.update(loss.item(), input_var.size(0))
        topframe.update(prec1[0], input_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, topframe=topframe))

    index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)

    index_matrix = torch.stack(index_matrix, dim=0).cuda(async=True).float()  # [21570]  --->  [380, 21570]

    output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]

    target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]

    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [380,21570] * [21570, 7] = [380,7]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

    prec_video = util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu(), topk=(1,))
    topVideo.update(prec_video[0], i + 1)
    print(' *Prec@Video {topVideo.avg:.3f}   *Prec@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))


def validate(val_loader, model):
    global record_
    batch_time = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    output_store_fc = []
    output_alpha    = []
    target_store = []
    index_vector = []
    with torch.no_grad():
        for i, (input_var, target, index) in enumerate(val_loader):
            # compute output
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input_var)
            ''' model & full_model'''
            f, alphas = model(input_var, phrase = 'eval')

            pred_score = 0
            output_store_fc.append(f)
            output_alpha.append(alphas)
            target_store.append(target)
            index_vector.append(index)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)

        index_matrix = torch.stack(index_matrix, dim=0).cuda(async=True).float()  # [21570]  --->  [380, 21570]

        output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
        output_alpha    = torch.cat(output_alpha, dim=0)     # [256,1] ... [256,1]  --->  [21570, 1]
        target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]

        ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
        weight_sourcefc = output_store_fc.mul(output_alpha)   #[21570,512] * [21570,1] --->[21570,512]
        sum_alpha = index_matrix.mm(output_alpha) # [380,21570] * [21570,1] -> [380,1]
        weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha)

        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

        if at_type == 'self-attention':
            pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
        if at_type == 'relation-attention':
            pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')

        prec_video = util.accuracy(pred_score.cpu(), target_vector.cpu(), topk=(1,))

        topVideo.update(prec_video[0], i + 1)
        print(' *Prec@Video {topVideo.avg:.3f} '.format(topVideo=topVideo))

        return topVideo.avg

if __name__ == '__main__':
    main()
