# 既往实验总结

1. 数据增强可以消除过拟合，但是数据增强也不能太多，不能乱加
    - 数据增强全加：陷入局部最优点
2. 损失函数的 lambda 设置很重要
    - self.lambda_xy = 0.05  # 预测框中心误差权重
    - self.lambda_wh = 0.05  # 预测框大小误差权重
    - self.lambda_class = 0.5  # 预测框类别误差权重
    - self.lambda_conf = 1.0  # 预测框置信度误差权重
    - self.lambda_obj = 1.0  # 预测框有物体掩码误差权重
    - self.lambda_noobj = 1.0  # 预测框无物体掩码误差权重
3. bs 没必要克制，和陷入鞍点影响不大，但是要合理设置 lr
4. 每轮迭代输出 map
5. 为什么 eval 的时候 batch_size 要设置的比 train 小

---

# 实验

## 初始实验

1. 无数据增强

```shell script
torch.multiprocessing.set_sharing_strategy('file_system')  # https://www.cnblogs.com/zhengbiqing/p/10478311.html
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # 0. 确保每次的伪随机数相同以便于问题的复现
    numpy.random.seed(0)
    torch.manual_seed(1)

    # 1. 训练参数
    Config = conf.config.VocConfig

    print("config:\n", Config)

    # 提示 OOM 或者显存不足请调小 Batch_size
    Freeze_Train_Batch_Size = 128
    Freeze_Eval_Batch_Size = 64

    Unfreeze_Train_Batch_Size = 32
    Unfreeze_Eval_Batch_Size = 16

    Init_Epoch = 0  # 起始世代
    Freeze_Epoch = 50  # 冻结训练的世代
    Unfreeze_Epoch = 2000  # 总训练世代

    Freeze_Epoch_LR = 1e-3
    Unfreeze_Epoch_LR = 1e-4

    # lr warm up
    Freeze_Epoch_Gamma = 0.96  # 0.96 ^ 50 = 0.12988
    Unfreeze_Epoch_Gamma = 0.97  # 0.97 ^ 50 = 0.22, 0.97 ^ 250 = 0.0005

    Num_Workers = 12
    Suffle = True

    Image_Set = "trainval"
    Validation_Split = 0.05  # 验证集大小

    Parallel = True
```

结果：

- Voc_Test_4_1_Epoch146-Train_Loss0.0054-Val_Loss24.6083.pth
- mAP(Voc_Test_4_1_Epoch146-Train_Loss0.0054-Val_Loss24.6083.pth)：

```shell script
5.41% = aeroplane AP 	||	score_threhold=0.5 : F1=0.07 ; Recall=4.26% ; Precision=28.17%
2.77% = bicycle AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.49% ; Precision=100.00%
1.02% = bird AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
0.59% = boat AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
0.03% = bottle AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
6.29% = bus AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.32% ; Precision=50.00%
12.16% = car AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.59% ; Precision=30.43%
12.93% = cat AP 	||	score_threhold=0.5 : F1=0.06 ; Recall=3.28% ; Precision=62.50%
5.21% = chair AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.69% ; Precision=83.33%
0.00% = cow AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
3.65% = diningtable AP 	||	score_threhold=0.5 : F1=0.04 ; Recall=2.14% ; Precision=34.78%
2.77% = dog AP 	||	score_threhold=0.5 : F1=0.00 ; Recall=0.13% ; Precision=33.33%
0.00% = horse AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
6.55% = motorbike AP 	||	score_threhold=0.5 : F1=0.05 ; Recall=2.40% ; Precision=45.00%
34.60% = person AP 	||	score_threhold=0.5 : F1=0.43 ; Recall=36.36% ; Precision=52.08%
0.04% = pottedplant AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
1.92% = sheep AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
3.58% = sofa AP 	||	score_threhold=0.5 : F1=0.01 ; Recall=0.75% ; Precision=37.50%
0.82% = train AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
1.28% = tvmonitor AP 	||	score_threhold=0.5 : F1=nan ; Recall=0.00% ; Precision=0.00%
mAP = 5.08%
```

---

## 数据增强实验 1

1. 添加全部数据增强
    - 位移 & 缩放
    - 颜色空间
    - 反转

---

## 数据增强实验 2

1. 数据增强多次切换

---

## 数据增强实验 3

1. 添加部分数据增强
    - 位移 & 缩放
    - 反转

---

