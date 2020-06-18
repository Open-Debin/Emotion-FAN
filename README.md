# Emotion-FAN.pytorch
 ICIP 2019: Frame Attention Networks for Facial Expression Recognition in Videos  [pdf](https://arxiv.org/pdf/1907.00193.pdf)
 
 [Debin Meng](michaeldbmeng19@outlook.com), [Xiaojiang Peng](https://pengxj.github.io/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), etc.
 
 If you are using pieces of the posted code, please cite the above paper.

## Visualization
We visualize the weights of attention module in the picture. The blue bars represent the ***self-attention weights*** and orange bars the ***final weights*** (the weights combine ***self-attention*** and ***relation-attention*** ).
<img width="450" height="245" src="https://github.com/DebinMeng19-OpenSourceLibrary/Emotion-FAN/blob/master/visualization_1.jpg"/><img width="400" height="245" src="https://github.com/DebinMeng19-OpenSourceLibrary/Emotion-FAN/blob/master/visualization_2.jpg"/>

Both weights can reflect the importance of frames. Comparing the blue and orange bars, the final weights of our FAN can assign higher weights to the more obvious face frames, while self-attention module could assign high weights on some obscure face frames. This explains why adding relation-attention boost performance.

## Citation
Whether or not it is useful to you, could you please do not hesitate to citing the paper. thanks, thanks, thanks.:
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
## Requirement
Pillow == 6.2.0

numpy == 1.17.2

torch == 1.3.0

torchvision == 0.4.1

## Download pretrain models
We share two **ResNet18** models, one model pretrained in **MS-Celeb-1M** and another one in **FER+**. [Baidu](https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg) or [OneDrive](https://1drv.ms/u/s!AhGc2vUv7IQtl1Pt7FhPXr_Kofd5?e=3MvPFX) 

Notice!!! The model trained on the AFEW dataset or CK+ dataset are not published.


## Demo AFEW
Training with self-attention
```
CUDA_VISIBLE_DEVICES=2 python Demo_AFEW_Attention.py --at_type 0
```
Training with self-attention and relation-attention
```
CUDA_VISIBLE_DEVICES=2 python Demo_AFEW_Attention.py --at_type 1
```
#### Options
* ``` --lr ```: initial learning rate
* ``` --at_type ```: 0 is self-attention; 1 is relation-attention
* ``` --epochs ```: number of total epochs to run
* ``` --momentum ```: momentum
* ``` --weight-decay ```: weight decay (default: 1e-4)
* ``` -e ```: evaluate model on validation set
* etc.

