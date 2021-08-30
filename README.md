# Self-Attention Recurrent Summarization Network with Reinforcement Learning for Video Summarization Task
___
IEEE International Conference on Multimedia and Expo (ICME) 2021


This is an implementation of DSR-RL (Deep Self-attention Recurrent summarization network with Reinforcement Learning).

## Datasets Preparation

Download the pre-processed datasets into `datasets/` folder, including [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), and [YouTube](https://sites.google.com/site/vsummsite/download) datasets.

```sh
mkdir -p datasets/ && cd datasets/
wget https://www.dropbox.com/s/8lvlf3knqwfeth9/datasets.zip
unzip datasets.zip
```

If the Dropbox link is unavailable to you, try downloading from below links.

+ (Baidu Cloud) Link: https://pan.baidu.com/s/18rZLlz14UATWzOod8CjRJQ Extraction Code: 17j2
+ (Google Drive) https://drive.google.com/file/d/1uqzcoOrBx9MEj4RUImFgVdal6TPK3UNl/view?usp=sharing

Now the datasets structure should look like

```
DSR-RL
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
```

## Userscore Preparation
For user score files can be download from [Summarizer](https://github.com/sylvainma/Summarizer) project.

+ (Google Drive) https://drive.google.com/drive/folders/1sbZZalh43n6fiSxWt_SIGgv72bt4rdoG

and put file `summarizer_dataset_summe_google_pool5` and `summarizer_dataset_tvsum_google_pool5.h5` in `datasets/` folder

```
DSR-RL
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    ├── summarizer_dataset_summe_google_pool5.h5
    ├── summarizer_dataset_tvsum_google_pool5.h5
```

### Training Model in Canonical Mode
#### SumMe
```
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s splits/summe_splits.json -m summe --gpu 0  --save-dir log/summe-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_summe_google_pool5.h5 --save-results
```

#### TVSum
```
python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s splits/tvsum_splits.json -m tvsum --gpu 0 --save-dir log/tvsum-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_tvsum_google_pool5.h5 --save-results
```

### Training Model in Augmented Mode

#### SumMe
```
python main.py -s datasets/splits/vasnet_summe_aug_splits.json -m summe --gpu 0 --save-dir log/summe-aug-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_summe_google_pool5.h5
```

#### TVSum
```
python main.py -s datasets/splits/vasnet_tvsum_aug_splits.json -m tvsum --gpu 0 --save-dir log/tvsum-aug-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_tvsum_google_pool5.h5
```

### Training Model in Transfer Mode

#### SumMe
```
python main.py -s datasets/splits/vasnet_summe_tran_splits.json -m summe --gpu 0 --save-dir log/summe-tran-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_summe_google_pool5.h5
```

#### TVSum
```
python main.py -s datasets/splits/vasnet_tvsum_tran_splits.json -m tvsum --gpu 0 --save-dir log/tvsum-tran-split --split-id 0 --verbose --rnn-cell gru --max-epoch 50 --userscore datasets/summarizer_dataset_tvsum_google_pool5.h5
```

## Citation
If this code is useful for you, please consider citing the paper as following detail:
```
@inproceedings{phaphuangwittayakul2021self,
  title={Self-Attention Recurrent Summarization Network with Reinforcement Learning for Video Summarization Task},
  author={Phaphuangwittayakul, Aniwat and Guo, Yi and Ying, Fangli and Xu, Wentian and Zheng, Zheng},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgement
This work is heavily based on [VASNet](https://github.com/ok1zjf/VASNet) and [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). Thanks them for their great work!

<em>DSR-RL</em> is freely available for non-commercial use. Don't hesitate to drop e-mail if you have any problem.
