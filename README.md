## End-to-End Speech Recognition on Pytorch
原项目 https://github.com/gentaiscool/end2end-asr-pytorch
基本同原开源项目
### Transformer-based Speech Recognition Model

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

If you use any source codes included in this toolkit in your work, please cite the following paper.
- Winata, G. I., Madotto, A., Wu, C. S., & Fung, P. (2019). Code-Switched Language Models Using Neural Based Synthetic Data from Parallel Sentences. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL) (pp. 271-280).
- Winata, G. I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung, P. (2019). Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer. arXiv preprint arXiv:1910.13923. (Accepted by ICASSP 2020)
- Zhou, S., Dong, L., Xu, S., & Xu, B. (2018). Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese. Proc. Interspeech 2018, 791-795.

### 亮点
- 支持多GPU 分布式训练
- 支持多数据集的训练和评估

### Requirements
- Python 3.5 or later
- Install Pytorch 1.4 (https://pytorch.org/)
- Install torchaudio (https://github.com/pytorch/audio)
- run ``❱❱❱ bash requirement.sh``

### 结果
AiShell-1
这是中文数据集，用字错率（Character Error Rate, CER）来衡量中文ASR效果好坏

| Decoding strategy | CER |
| ------------- | ------------- |
| Greedy | 14.5% |
| Beam-search (beam width=8) | 13.5% |

### 数据处理
#### AiShell-1 (Chinese)
先从 https://www.openslr.org/33/.下载数据

有两个数据文件夹  resource_aishell  包含播音人信息，和主要拼音标注

                  data_aishell 包含主要数据集和文本标注


需要先将wav 文件夹的声波数据进行划分为 train  dev 和test 文件夹

这里更改了位置位于 Aishell_dataset 的 transcript 文件夹下

通过split.py  将标签放在各个文件的目录下

结构如下 ：

Aishell_dataset/
<<<<<<< HEAD
```
=======

>>>>>>> 32c3addddaa4008f84849cb3da4e8acbccb2010f
├── transcript
│   ├── train
│   │   ├── S0724
│   │   │   ├── BAC009S0724W0121.docx
│   │   │   ├── BAC009S0724W0121.wav
.......
│   ├──  dev
│   │   ├── S0723
│   │   │   ├── BAC009S0723W0122.docx
│   │   │   ├── BAC009S0723W0122.wav
.........
│   ├── test
│   │   ├── S0725
│   │   │   ├── BAC009S0725W0124.docx
│   │   │   ├── BAC009S0725W0124.wav
.........
<<<<<<< HEAD

```
=======
>>>>>>> 32c3addddaa4008f84849cb3da4e8acbccb2010f
docx为每个wav文件对应的标签文件
然后运行数据预处理脚本，生成相应清洗和加语言标签的标签文件夹
       transcript_clean
       transcript_clean_lang

```console
❱❱❱ python data/aishell.py
```
同时会生成manifest ，即数据和标签对应关系

<<<<<<< HEAD
#### Librispeech  英文数据集略
=======
#### Librispeech  英文数据集略  
>>>>>>> 32c3addddaa4008f84849cb3da4e8acbccb2010f
To automatically download the data
```console
❱❱❱ python data/librispeech.py
```

### 训练
```console
usage: train.py [-h] [--train-manifest-list] [--valid-manifest-list] [--test-manifest-list] [--cuda] [--verbose] [--batch-size] [--labels-path] [--lr] [--name] [--save-folder] [--save-every] [--feat_extractor] [--emb_trg_sharing] [--shuffle] [--sample_rate] [--label-smoothing] [--window-size] [--window-stride] [--window] [--epochs]  [--src-max-len] [--tgt-max-len] [--warmup] [--momentum] [--lr-anneal] [--num-layers] [--num-heads] [--dim-model] [--dim-key] [--dim-value] [--dim-input] [--dim-inner] [--dim-emb] [--shuffle]
```
#### Parameters
```
- feat_extractor: "emb_cnn" or "vgg_cnn" as the feature extractor, or set "" for none
    - emb_cnn: add 4-layer 2D CNN
    - vgg_cnn: add 6-layer 2D CNN
- cuda: train on GPU
- shuffle: randomly shuffle every batch
```

#### Example
```console
❱❱❱ python train.py --train-manifest-list data/manifests/aishell_train_manifest.csv --valid-manifest-list data/manifests/aishell_dev_manifest.csv --test-manifest-list data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 1e-4 --name aishell_drop0.1_cnn_batch12_4_vgg_layer4 --save-folder save/ --save-every 5 --feat_extractor vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle --min-lr 1e-6 --k-lr 1
```
Use ``python train.py --help`` for more parameters and options.

#### 结果
##### AiShell-1 Loss Curve
<img src="img/aishell_loss.jpg"/>

### 多GPU训练
```
usage: train.py [--parallel] [--device-ids]
```

#### 参数
```
- parallel: split batches to GPUs (the number of batch has to be divisible by the number of GPUs)
- device-ids: GPU ids
```

#### Example
```console
❱❱❱ CUDA_VISIBLE_DEVICES=0,1 python train.py --train-manifest-list data/manifests/aishell_train_manifest.csv --valid-manifest-list data/manifests/aishell_dev_manifest.csv --test-manifest-list data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 1e-4 --name aishell_drop0.1_cnn_batch12_4_vgg_layer4 --save-folder save/ --save-every 5 --feat_extractor vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle --min-lr 1e-6 --k-lr 1 --parallel --device-ids 0 1
```
### 测试
```
usage: test.py [-h] [--test-manifest] [--cuda] [--verbose] [--continue_from]
```
#### 参数
```
- cuda: test on GPU
- continue_from: path to the trained model
```
#### Example
```console
❱❱❱ python test.py --test-manifest-list libri_test_clean_manifest.csv --cuda --continue_from save/model
```

Use ``python multi_train.py --help`` for more parameters and options.

### 自定义数据集
#### 信息文件
按照下面的格式建立CSV 格式manifest 信息文件 :
```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
每行包括语音文件路径和语言标签路径   .

#### 标签文件
You need to specify all characters in the corpus by using the following JSON format:
```
[ 
  "_",
  "'",
  "A",
  ...,
  "Z",
  " "
]
```

