# Emotion-FAN.pytorch
 ICIP 2019: Frame Attention Networks for Facial Expression Recognition in Videos  [pdf](https://arxiv.org/pdf/1907.00193.pdf)
 
 [Debin Meng](michaeldbmeng19@outlook.com), [Xiaojiang Peng](https://pengxj.github.io/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), etc.

## Citation
If you are using pieces of the posted code, please cite the above paper. thanks.:
```
@inproceedings{meng2019frame,
  title={frame attention networks for facial expression recognition in videos},
  author={Meng, Debin and Peng, Xiaojiang and Wang, Kai and Qiao, Yu},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={3866--3870},
  year={2019},
  organization={IEEE},
  url={https://github.com/Open-Debin/Emotion-FAN}
}
```
## User instructions
This is the latest version. You can find the old version in branch [version1.0_august_2019](https://github.com/Open-Debin/Emotion-FAN/tree/version1.0_august_2019) or tag [v1.0_august_2019](https://github.com/Open-Debin/Emotion-FAN/releases/tag/v1.0_august_2019).

step1: [Install Dependencies](#dependencies)

step2: [Download Pretrain Model and Dataset](#download-pretrain-models-and-published-dataset)

step3: [Face Alignment](#face-alignment)

step4: [Running Experiments](#running-experiments)

## Visualization
We visualize the weights of attention module in the picture. The blue bars represent the ***self-attention weights*** and orange bars the ***final weights*** (the weights combine ***self-attention*** and ***relation-attention*** ).

<img width="450" height="245" src="https://github.com/Open-Debin/Emotion-FAN/blob/master/github_materials/visualization_1.jpg"/><img width="400" height="245" src="https://github.com/Open-Debin/Emotion-FAN/blob/master/github_materials/visualization_2.jpg"/>

Both weights can reflect the importance of frames. Comparing the blue and orange bars, the final weights of our FAN can assign higher weights to the more obvious face frames, while self-attention module could assign high weights on some obscure face frames. This explains why adding relation-attention boost performance.

### dependencies
```
# create the environment for the project
conda create -n emotion_fan python=3.9
conda activate emotion_fan

# install ffmpeg
sudo apt-get update 
sudo apt-get install ffmpeg

# install dlib
sudo apt-get update
sudo apt-get install cmake
sudo apt-get install libboost-python-dev
pip3 install dlib

# install cv2
pip install opencv-python
```
install [pytorch](https://pytorch.org/get-started/locally/)

### download pretrain models and published dataset
We share two **ResNet18** models, one model pretrained in **MS-Celeb-1M** and another one in **FER+**. [Baidu](https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg) or [OneDrive](https://1drv.ms/u/s!AhGc2vUv7IQtl1Pt7FhPXr_Kofd5?e=3MvPFX) . Please put the model at the directory: ***"Emotion-FAN/pretrain_model/"***. 

You can get the AFEW dataset by ask the official organizer: shreya.ghosh@iitrpr.ac.in and emotiw2014@gmail.com . Also, you can get the [ck+ dataset](http://www.jeffcohn.net/Resources/). Please unzip the ***train (val)*** part of AFEW dataset at the directory: ***"./Emotion-FAN/data/video/train_afew (val_afew)"***, put the file ***"cohn-kanade-images"*** of the [ck+ dataset](http://www.jeffcohn.net/Resources/) at the directory: ***"./Emotion-FAN/data/frame/"*** .

### face alignment
#### AFEW Dataset
```
cd ./data/face_alignment_code/
python video2frame_afew.py
python frame2face_afew.py
```
#### CK+ Dataset
```
cd ./data/face_alignment_code/
python frame2face_ck_plus.py
```

### running experiments
#### AFEW Dataset <br>
```
# Baseline
CUDA_VISIBLE_DEVICES=0 python baseline_afew.py
# Training with self-attention
CUDA_VISIBLE_DEVICES=0 python fan_afew_traintest.py --at_type 0
# Training with relation-attention
CUDA_VISIBLE_DEVICES=0 python fan_afew_traintest.py --at_type 1
```
#### CK+ Dataset <br>
```
# Baseline. Notice you should test on fold 1,2, ..., 10. And finally average performance of the ten folds.
CUDA_VISIBLE_DEVICES=0 python baseline_ck_plus.py --fold 10
# Training with self-attention
CUDA_VISIBLE_DEVICES=0 python fan_ckplus_traintest.py --at_type 0
# Training with relation-attention
CUDA_VISIBLE_DEVICES=0 python fan_ckplus_traintest.py --at_type 1
```
#### Options
* ``` --lr ```: initial learning rate
* ``` --at_type ```: 0 is self-attention; 1 is relation-attention
* ``` --epochs ```: number of total epochs to run
* ``` --fold ```: (only use for ck+) which fold used for test in ck+
* ``` -e ```: evaluate model on validation set
* etc.

