import numpy as np
import torch
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.optim as op
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from sample_data.pytorch_handbook.chapter7.layers.modules import MultiBoxLoss
from sample_data.pytorch_handbook.chapter7.ssd import build_ssd
from sample_data.pytorch_handbook.chapter7.data import *
from sample_data.pytorch_handbook.chapter7.utils.augmentations import *

logging.basicConfig(level=logging.INFO)

"""
事前学習モデルを再学習させ新たなSSDモデルを作成する。
⇒ モデル評価時には既存のデータセットを用いるのが一般的
   例　・PASCAL VOC(PASCAL Visual Object Classes)：アノテーション付きの画像データセット
   　　・COCO(Common Object in Context)：セマンティックセグメンテーション情報
        (いわゆるアノテーション／ラベリングよりも詳しい、画素レベルでの物体認識情報)が付加されたデータセット

⇒ 今回用いる「BCCDデータセット」はPASCAL VOC仕様である

"""

weight_path = "src/sample_data/pytorch_handbook/chapter7/weights/"
args = {'dataset':'BCCD',  # VOC → BCCD
    'basenet':'vgg16_reducedfc.pth',
    'batch_size':8,
    'resume':'',
    'start_iter':0,
    'num_workers':0,  # 4 → 0
    'cuda':True,  # Macの場合False
    'lr':5e-4,
    'momentum':0.9,
    'weight_decay':5e-4,
    'gamma':0.1,
    'save_folder': weight_path
    }

# 学習率の調整
def adjust_learning_rate(optimizer, gamma, step):
    lr = args['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
# xavierの初期値
def xavier(param):
    init.xavier_uniform_(param)
 
# 重みの初期化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def train(cfg, network, dataset, optimizer, criterion):
    network.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    epoch_size = len(dataset) // args['batch_size']
    step_index = 0

    batch_iterator = None
    for iteration in range(args["start_iter"], cfg["max_iter"]+1):
        logging.info(f'start {iteration}th iteration !!')
        if (not batch_iterator) or (iteration % epoch_size ==0):
            batch_iterator = iter(dataset)
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args['gamma'], step_index)
        
        # バッチサイズ分の訓練データをload(for文で回してもよい)
        images, targets = next(batch_iterator)
        # forward
        output = network(images)
        # backprop
        optimizer.zero_grad() # 勾配の初期化
        loss_l, loss_c = criterion(output, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        logging.info(f"Loss : {loss.item()}")
        
        
def main():
    # 訓練データセットの呼び出し
    cfg = voc
    dataset = VOCDetection(root=VOC_ROOT,transform=SSDAugmentation(cfg['min_dim'],MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    network = ssd_net

    # 事前学習モデルのパラメータを取得する
    if args['resume']:
        print('Resuming training, loading {}...'.format(args['resume']))
        ssd_net.load_weights(args['resume'])
    else:
        vgg_weights = torch.load(args['save_folder'] + args['basenet'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    # 新しく追加するレイヤの重みをxavierで初期化する
    if not args['resume']: # args['resume']は""(空文字)
        print('Initializing weights...')
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
    
    # 最適化パラメータの設定
    optimizer = op.SGD(network.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
 
    # 損失関数の設定
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False)

    data_loader = DataLoader(
        dataset, 
        batch_size=args["batch_size"], 
        num_workers=args["num_workers"], 
        shuffle=True, 
        collate_fn=detection_collate, # サンプルのリストをマージして、Tensorのミニバッチを形成。マップスタイルのデータセットから一括読み込み時使用。
        pin_memory=True
    )
    model = train(cfg, network, data_loader, optimizer, criterion) 


 

if __name__ == "__main__":
    main() 