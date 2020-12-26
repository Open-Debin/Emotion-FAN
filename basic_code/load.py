from __future__ import print_function
import torch
print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
from basic_code import data_generator

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},

              'AFEW':{0: 'Happy',1: 'Angry',2: 'Disgust',3: 'Fear',4: 'Sad',5: 'Neutral',6: 'Surprise',
                  'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Neutral': 5,'Sad': 4,'Surprise': 6}}

def ckplus_faces_baseline(video_root, video_list, fold, batchsize_train, batchsize_eval):
    train_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='train'
                                        )

    val_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='test'
                                        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def ckplus_faces_fan(video_root, video_list, fold, batchsize_train, batchsize_eval):
    train_dataset = data_generator.TenFold_TripleImageDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([
                                            transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='train',
                                        )

    val_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='test'
                                        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def afew_faces_baseline(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.VideoDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader

def afew_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, val_loader


def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
