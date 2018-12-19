# Deep Video Frame Interpolation using Cyclic Frame Generation
Video frame interpolation algorithms predict intermediate frames to produce videos with higher frame rates and smooth view transitions given two consecutive frames as inputs. We propose that: synthesized frames are more reliable if they can be used to reconstruct the input frames with high quality. Based on this idea, we introduce a new loss term, the cycle consistency loss. The cycle consistency loss can better utilize the training data to not only enhance the interpolation results, but also maintain the performance better with less training data. It can be integrated into any frame interpolation network and trained in an end-to-end manner. In addition to the cycle consistency loss, we propose two extensions: motion linearity loss and edge-guided training. The motion linearity loss approximates the motion between two input frames to be linear and regularizes the training. By applying edge-guided training, we further improve results by integrating edge information into training. Both qualitative and quantitative experiments demonstrate that our model outperforms the state-of-the-art methods.

[[Project]](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/CyclicGen) [Paper]

## Overview
This is the author's reference implementation of the video frame interpolation using TensorFlow described in:
"Deep Video Frame Interpolation using Cyclic Frame Generation"
[Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/), [Yi-Tung Liao](http://www.cmlab.csie.ntu.edu.tw/~queenieliaw/), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/), [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/) (Academia Sinica & National Taiwan University & MediaTek)
in 33rd AAAI Conference on Artificial Intelligence (AAAI) 2019, Oral Presentation
Should you be making use of our work, please cite our paper [1].

<img src='./teaser.png' width=400>

Further information please contact [Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/).

## Requirements setup
* [TensorFlow](https://www.tensorflow.org/)

## Data Preparation
* [Deep Voxel Flow (DVF)](https://github.com/liuziwei7/voxel-flow)

## Getting started
* Run the training script:
``` bash
python3 CyclicGen_train_stage1.py --subset=train
```
* Run the testing and evaluation script:
``` bash
python3 CyclicGen_train_stage1.py --subset=test
```

## Citation
```
[1]  @inproceedings{liu2019cyclicgen,
         author = {Yu-Lun Liu, Yi-Tung Liao, Yen-Yu Lin, and Yung-Yu Chuang},
         title = {Deep Video Frame Interpolation using Cyclic Frame Generation},
         booktitle = {AAAI},
         year = {2019}
}
```
