import os
import time
import torch
from pathlib import Path

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (256 * 1) -> (1 * 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, at_type=''):

    if not os.path.exists('./model'):
        os.makedirs('./model')

    epoch = state['epoch']
    save_dir = './model/'+at_type+'_' + str(epoch) + '_' + str(round(float(state['accuracy']), 4))
    torch.save(state, save_dir)
    print(save_dir)
    
def time_now():
  ISOTIMEFORMAT='%d-%h-%Y-%H-%M-%S'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

class Logger(object):
    def __init__(self, log_dir, title, args=False):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
        self.title = title
        self.log_file = '{:}/{:}_date_{:}.txt'.format(self.log_dir,title, time_now())
        self.file_writer = open(self.log_file, 'a')
        
        if args:
            for key, value in vars(args).items():
                self.print('  [{:18s}] : {:}'.format(key, value))
        self.print('{:} --- args ---'.format(time_now()))
        
    def print(self, string, fprint=True, is_pp=False):
        if is_pp: pp.pprint (string)
        else:     print(string)
        if fprint:
          self.file_writer.write('{:}\n'.format(string))
          self.file_writer.flush()
            
    def write(self, string):
        self.file_writer.write('{:}\n'.format(string))
        self.file_writer.flush()  
        