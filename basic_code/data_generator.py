#coding=utf-8
import pdb
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

## data generator for afew
class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None, csv = False):

        self.imgs_first, self.index = load_imgs_total_frame(video_root, video_list, rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

# 
class TripleImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index = load_imgs_tsn(video_root, video_list,
                                                                                           rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)
        return img_first, img_second, img_third, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

def load_imgs_tsn(video_root, video_list, rectify_label):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            num_per_part = int(img_count) // 3

            if int(img_count) > 3:
                for i in range(img_count):

                    random_select_first = random.randint(0, num_per_part)
                    random_select_second = random.randint(num_per_part, num_per_part * 2)
                    random_select_third = random.randint(2 * num_per_part, len(img_lists) - 1)

                    img_path_first = os.path.join(video_path, img_lists[random_select_first])
                    img_path_second = os.path.join(video_path, img_lists[random_select_second])
                    img_path_third = os.path.join(video_path, img_lists[random_select_third])

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            else:
                for j in range(len(img_lists)):
                    img_path_first = os.path.join(video_path, img_lists[j])
                    img_path_second = os.path.join(video_path, random.choice(img_lists))
                    img_path_third = os.path.join(video_path, random.choice(img_lists))

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            ###  return video frame index  #####
            index.append(np.ones(img_count) * id)  # id: 0 : 379
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, imgs_second, imgs_third, index


def load_imgs_total_frame(video_root, video_list, rectify_label):
    imgs_first = list()

    with open(video_list, 'r') as imf:
        index = []
        video_names = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video

            for frame in img_lists:
                # pdb.set_trace()
                imgs_first.append((os.path.join(video_path, frame), label))
            ###  return video frame index  #####
            video_names.append(video_name)
            index.append(np.ones(img_count) * id)
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, index
    
## data generator for ck_plus
class TenFold_VideoDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', rectify_label=None, transform=None, fold=1, run_type='train'):
        self.imgs_first, self.index = load_imgs_tenfold_totalframe(video_root, video_list, rectify_label, fold, run_type)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert('RGB')
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

class TenFold_TripleImageDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', rectify_label=None, transform=None, fold=1, run_type='train'):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index = load_imgs_tsn_tenfold(video_root,video_list,rectify_label, fold, run_type)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)

        return img_first, img_second, img_third, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


def load_imgs_tenfold_totalframe(video_root, video_list, rectify_label, fold, run_type):
    imgs_first = list()
    new_imf = list()

    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:
        imf = imf.readlines()
    if run_type == 'train':
        fold_ = list(range(1, 11))
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9, 10] -> [2,3,4,5,6,7,8,9,10]

        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold
            for index, item in enumerate(
                    imf):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    if run_type == 'test':
        fold_ = fold
        fold_str = str(fold_) + '-fold'
        for index, item in enumerate(imf):
            if fold_str in item:
                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])

    index = []
    for id, line in enumerate(new_imf):

        video_label = line.strip().split()

        video_name = video_label[0]  # name of video
        try:
            label = rectify_label[video_label[1]]  # label of video
        except:
            pdb.set_trace()
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        ###  for sampling triple imgs in the single video_path  ####
        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending
        
        img_lists = img_lists[ - int(round(len(img_lists))) : ]

        img_count = len(img_lists)  # number of frames in video
        for frame in img_lists:
            imgs_first.append((os.path.join(video_path, frame), label))
        ###  return video frame index  #####
        index.append(np.ones(img_count) * id)

    index = np.concatenate(index, axis=0)
    return imgs_first, index

def load_imgs_tsn_tenfold(video_root, video_list, rectify_label, fold, run_type):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    new_imf = list()
    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:
        imf = imf.readlines()
    if run_type == 'train':
        fold_ = list(range(1, 11))
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9,10] -> [2,3,4,5,6,7,8,9,10]
        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold
            for index, item in enumerate(
                    imf):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'
    if run_type == 'test':
        fold_ = fold
        fold_str = str(fold_) + '-fold'
        for index, item in enumerate(imf):
            if fold_str in item:
                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])
    ''' Make triple-image list '''
    index = []
    for id, line in enumerate(new_imf):
        video_label = line.strip().split()
        video_name = video_label[0]  # name of video
        label = rectify_label[video_label[1]]  # label of video
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        ###  for sampling triple imgs in the single video_path  ####
        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending
        img_lists = img_lists[ - int(round(len(img_lists))):]
        img_count = len(img_lists)  # number of frames in video
        num_per_part = int(img_count) // 5
        if int(img_count) > 5:
            for i in range(img_count):
                # pdb.set_trace()
                random_select_first = random.randint(0, num_per_part)
                random_select_second = random.randint(num_per_part, 2 * num_per_part)
                random_select_third = random.randint(2 * num_per_part, 3 * num_per_part)

                img_path_first = os.path.join(video_path, img_lists[random_select_first])
                img_path_second = os.path.join(video_path, img_lists[random_select_second])
                img_path_third = os.path.join(video_path, img_lists[random_select_third])

                imgs_first.append((img_path_first, label))
                imgs_second.append((img_path_second, label))
                imgs_third.append((img_path_third, label))

        else:
            for j in range(len(img_lists)):
                img_path_first = os.path.join(video_path, img_lists[j])
                img_path_second = os.path.join(video_path, random.choice(img_lists))
                img_path_third = os.path.join(video_path, random.choice(img_lists))

                imgs_first.append((img_path_first, label))
                imgs_second.append((img_path_second, label))
                imgs_third.append((img_path_third, label))

        ###  return video frame index  #####
        index.append(np.ones(img_count) * id)  # id: 0 : 379
    index = np.concatenate(index, axis=0)
    # index = index.astype(int)
    # pdb.set_trace()
    return imgs_first, imgs_second, imgs_third, index