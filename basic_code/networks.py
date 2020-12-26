import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

###''' self-attention; relation-attention '''

class ResNet_AT(nn.Module):
    def __init__(self, block, layers, num_classes=1000, end2end=True, at_type=''):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet_AT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)
        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())

        self.pred_fc1 = nn.Linear(512, 7)
        self.pred_fc2 = nn.Linear(1024, 7)
        self.at_type = at_type

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', phrase='train', AT_level='first_level',vectors='',vm='',alphas_from1='',index_matrix=''):

        vs = []
        alphas = []

        assert phrase == 'train' or phrase == 'eval'
        assert AT_level == 'first_level' or AT_level == 'second_level' or AT_level == 'pred'
        if phrase == 'train':
            num_pair = 3

            for i in range(num_pair):
                f = x[:, :, :, :, i]  # x[128,3,224,224]

                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)

                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]

                # MN_MODEL(first Level)
                vs.append(f)
                alphas.append(self.alpha(self.dropout(f)))

            vs_stack = torch.stack(vs, dim=2)
            alphas_stack = torch.stack(alphas, dim=2)

            if self.at_type == 'self-attention':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            if self.at_type == 'self_relation-attention':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
                betas = []
                for i in range(len(vs)):
                    vs[i] = torch.cat([vs[i], vm1], dim=1)
                    betas.append(self.beta(self.dropout(vs[i])))

                cascadeVs_stack = torch.stack(vs, dim=2)
                betas_stack = torch.stack(betas, dim=2)
                output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))

            if self.at_type == 'self-attention':
                vm1 = self.dropout(vm1)
                pred_score = self.pred_fc1(vm1)

            if self.at_type == 'self_relation-attention':
                output = self.dropout2(output)
                pred_score = self.pred_fc2(output)

            return pred_score

        if phrase == 'eval':
            if AT_level == 'first_level':
                f = self.conv1(x)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)

                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]
                # MN_MODEL(first Level)
                alphas = self.alpha(self.dropout(f))

                return f, alphas

            if AT_level == 'second_level':
                assert self.at_type == 'self_relation-attention'
                vms = index_matrix.permute(1, 0).mm(vm)  # [381, 21783] -> [21783,381] * [381,512] --> [21783, 512]
                vs_cate = torch.cat([vectors, vms], dim=1)

                betas = self.beta(self.dropout(vs_cate))
                ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
                ''' alpha * beta '''
                weight_catefc = vs_cate.mul(alphas_from1)  # [21570,512] * [21570,1] --->[21570,512]
                alpha_beta = alphas_from1.mul(betas)
                sum_alphabetas = index_matrix.mm(alpha_beta)  # [380,21570] * [21570,1] -> [380,1]
                weightmean_catefc = index_matrix.mm(weight_catefc).div(sum_alphabetas)

                weightmean_catefc = self.dropout2(weightmean_catefc)
                pred_score = self.pred_fc2(weightmean_catefc)

                return pred_score

            if AT_level == 'pred':
                if self.at_type == 'self-attention':
                    pred_score = self.pred_fc1(self.dropout(vm))

                return pred_score

''' self-attention; relation-attention '''
def resnet18_at(pretrained=False, **kwargs):
    # Constructs base a ResNet-18 model.
    model = ResNet_AT(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
