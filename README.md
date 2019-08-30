# Emotion-FAN.pytorch
 ICIP 2019: Frame Attention Networks for Facial Expression Recognition in Videos  [pdf](https://arxiv.org/pdf/1907.00193.pdf)
 
 [Debin Meng](michaeldbmeng19@outlook.com), [Xiaojiang Peng](https://pengxj.github.io/), etc.
 
## Download pretrain models
We share two model of **ResNet18**, include a model pretrained in **MS-Celeb-1M** and another in **FER+**.

[Baidu](https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg) or [Dropbox](https://github.com/DebinMeng19-OpenSourceLibrary/Emotion-FAN/blob/master/README.md)


## Demo AFEW
Training with self-attention
```
CUDA_VISIBLE_DEVICES=2 python Demo_AFEW_Attention.py --at_type 0
```
Training with relation-attention
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

## Visualization
<img width="250" height="150" src="https://github.com/DebinMeng19-OpenSourceLibrary/Emotion-FAN/blob/master/visualization1.png"/>
<img width="250" height="150" src="https://github.com/DebinMeng19-OpenSourceLibrary/Emotion-FAN/blob/master/visualization2.png"/>

## Citation
If you find this code useful in your research, please consider citing us:
```
@misc{1907.00193,
Author = {Debin Meng and Xiaojiang Peng and Kai Wang and Yu Qiao},
Title = {frame attention networks for facial expression recognition in videos},
Year = {2019},
Eprint = {arXiv:1907.00193},
}
```
