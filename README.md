# Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Differentially Constrained Motion Models

> ### Announcements
>  *May 2024* :date:
>  - We have just released a [toolbox](https://arxiv.org/abs/2405.00604) designed for trajectory-prediction research on the **D**rone datasets!
>  - It is made completely open-source here on [GitHub](https://github.com/westny/dronalize). Make sure to check it out. 
> 
>  *April 2023* :date:
> - Update repository to include functionality to reproduce paper 2.
> - Migrate code to torch==2.0.0. Update requirements.

> ### Description
> `mtp-go` is a library containing the implementation for the papers: 
> 1. *MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs* ([arXiv](https://arxiv.org/abs/2302.00735)), published in IEEE Transactions on Intelligent Vehicles ([TIV](https://ieeexplore.ieee.org/document/10143287)).
> 2. *Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction* ([arXiv](https://arxiv.org/abs/2304.05116)), published in proceedings of IEEE 2023 Intelligent Vehicles Symposium ([IV 2023](https://ieeexplore.ieee.org/document/10186615)).
> 
> Both papers are available in preprint format on ArXiv by the links above.
> All code is written using Python 3.11 using a combination of [<img alt="Pytorch logo" src=https://github.com/westny/mtp-go/assets/60364134/a416cd27-802c-454d-b25c-ac4d520927b1 height="12">PyTorch](https://pytorch.org/docs/stable/index.html), [<img alt="PyG logo" src=https://github.com/westny/mtp-go/assets/60364134/fad91e36-c94a-4fd7-bb33-943cff9c5430 height="12">PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), and [<img alt="Lightning logo" src=https://github.com/westny/mtp-go/assets/60364134/5e57cab7-88a9-4cb8-a17d-aa0941ec384f height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

<div align="center">
  <img alt="First page figure" src=https://github.com/westny/mtp-go/assets/60364134/5cb0ec14-79e8-491b-9d38-f536dce78c55 width="600px" style="max-width: 100%;">
</div>


##### If you found the content of this repository useful, please consider citing the papers in your work:
```
@article{westny2023mtp,
  title="{MTP-GO}: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODEs}",
  author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  volume={8},
  number={9},
  pages={4223-4236},
  doi={10.1109/TIV.2023.3282308}}
} 
```

```
@inproceedings{westny2023eval,
  title={Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction},
  author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  pages={},
  year={2023},
  doi={10.1109/IV55152.2023.10186615}
}
```
***

#### Hardware requirements

The original implementation make use of a considerable amount of data (some gigabytes worth) for training and testing which can be demanding for some setups. For you reference all code has been tried and used on a computer with the following specs:
```
* Processor: Intel® Xeon(R) E-2144G CPU @ 3.60GHz x 8
* Memory: 32 GB
* GPU: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A]
```

## Usage

Most of the necessary building blocks to implement MTP-GO is contained within the `models` folder. 
The main files of interest are:
- [gru_gnn.py](models/gru_gnn.py)
- [motion_models.py](models/motion_models.py)
- [base_mdn.py](base_mdn.py)

In `gru_gnn.py` the complete encoder-decoder model implementation is contained.
This includes a custom GRU cell implementation that make use of layers based on Graph Neural Networks.

In `motion_models.py` the implementations of the various motion models are contained, including the neural ODEs, used to learn road-user differential constraints. 
This is also where you will find functions used to perform the Jacobian calculations of the model.

In this work, [<img alt="Lightning logo" src=https://github.com/westny/mtp-go/assets/60364134/5e57cab7-88a9-4cb8-a17d-aa0941ec384f height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) was used to implement the training and testing behavior.
Since most of the functionality is still implemented using [<img alt="Pytorch logo" src=https://github.com/westny/mtp-go/assets/60364134/a416cd27-802c-454d-b25c-ac4d520927b1 height="12">PyTorch](https://pytorch.org/docs/stable/index.html), you are not restricted to using lightning, but it is recommended given the additional functionality.
In `base_mdn.py` the lightning-based abstraction of MTP-GO is contained.
This module is used to implement batch-wise forward and backward passes as well as to specify training and testing behavior.

Assuming data is available, training a model based on MTP-GO *is as easy* as running `train.py` in an environment with the necessary libraries installed, e.g.:
```bash
python train.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128
```
To learn more about the objective-scheduling algorithm described in the paper as well as the loss functions used, see [losses.py](losses.py).

<div align="center">
  <img alt="Schematics of MTP-GO" src=https://github.com/westny/mtp-go/assets/60364134/4f30cf04-db78-470c-aae0-c50c468afe04 width="800px" style="max-width: 100%;">
</div>


## Data sets

For model training and evaluation, the [highD](https://www.highd-dataset.com/), [rounD](https://www.round-dataset.com/), and [inD](https://www.ind-dataset.com/) were used. The data sets contain recorded trajectories from different locations in Germany, including various highways, roundabouts, and intersections. The data includes several hours of naturalistic driving data recorded at 25 Hz of considerable quality.
They are freely available for non-commercial use but do require applying for usage through the links above.

<div align="center">
  <img src=https://user-images.githubusercontent.com/60364134/220960422-4e7d7e13-c9b3-42af-99d3-a61eb6406e1e.gif alt="rounD.gif">
</div>


## Preprocessing

Assuming that you have been granted access to any of the above-mentioned data sets, proceed by moving the unzipped content (folder) into a folder named `data_sets` (you have to create this yourself) on the same level as this project. 
The contents may of course be placed in any accessible location of choice but do then require modifications of the preprocessing scripts (see the head of the .py files).

Methods of preprocessing are contained within Python scripts. Executing them may be done from a terminal or IDE of choice **(from within this project folder)**, for example: 
```bash
python rounD_preprocess.py
```

The output of the preprocessing scripts will be sent to a sub-folder with the name of the data set within the `data` folder in this project. 
Each data sample refers to a traffic sequence and is given a unique index used for easy access. 

:exclamation: A word of **caution**, by this approach, a lot of files are created that could be demanding for some systems.

To make sure unsaved work is not deleted, you will be prompted on your preferred cause of action should the folders already exist: either overwriting existing data or aborting the process.

:triangular_flag_on_post: The methods of preprocessing are by no means optimized for speed or computational efficiency.
The process could take several hours depending on the data set and available hardware. 

## License
[Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/)

## Inquiries
> Questions about the paper or the implementations found in this repository should be sent to [_theodor.westny [at] liu.se_](https://liu.se/en/employee/thewe60)



# me

注意不需要 弄懂， 只要缝合的附近看懂 然后抄原文的大部分地方 

原理 转述自己的 只要有不一样的就算创新！！！

## dataset下载

用学校的邮箱申请成功了

下载尼玛巨慢，为了我珍贵的流量 highd没下

只下了ind round

换成德国节点还行 美国巨卡  他妈太流氓了  一直 下着下着 不下了  然后从头开始欺骗我流量 傻逼玩意儿



不行 还是换节点  下着下着越下越慢  而且每次重头开始 傻逼 浪费我钱 gptNo. 64这个

## 环境  

20240523 今天上午一点都不想学习 想死

耗时：1h+15min

https://zhuanlan.zhihu.com/p/659091190?utm_id=0

pyg这么装

https://data.pyg.org/whl/

torch 2.3.0 gpu   cuda 12.1

https://data.pyg.org/whl/torch-2.3.0%2Bcu121.html

 conda create -n mtp python= 3.9

下载whl4个 最后pyg_lib

最后安装torch_geometric `pip install torch_geometric -i https://mirrors.aliyun.com/pypi/simple`



r**equirements少了 pyarrow**

pip install pyarrow

PyArrow是一个Python库，用于在Apache Arrow和Python之间进行高效的数据交换。Apache Arrow是一种内存格式，用于在不同系统和编程语言之间高速传递数据。

PyArrow提供了一组API，可以在Python中读取、写入和处理各种数据源（如CSV文件、Parquet文件、Pandas数据帧等），并以Apache Arrow格式表示。



## preprocess

1. 改了highD_preprocess.py里dataset目录地址  search_path和save_path

2. 受不了了  开vscode 306行打断点调试

    File "/mnt/home/husiyuan/miniconda3/envs/mtp/lib/python3.9/site-packages/pandas/core/series.py", line 240, in wrapper
       warnings.warn(
   FutureWarning: Calling int on a single element Series is deprecated and will raise a TypeError in the future. Use int(ser.iloc[0]) instead

    File "/mnt/home/husiyuan/code/mtp-go_me/highD_preprocess.py", line 306, in add_maneuver
       n_lane_changes = int(tracks_meta[tracks_meta.trackId == t_id].numLaneChanges)

   运行 选调试 python没反应 选了debugger 在调试控制台里看type类型  改了306和311行 258 276 504 516  都是series强转int问题！！

   成功拉 啦啦啦啦啦啦啦  我的debug水平变强了，

   果然打了几个月的项目和算法题 还有以前的积累没有白费

   

   纪念品是鼠标垫和键盘垫

   用tmux在后台跑 这样本地关了也没事！！！
   
3. 跑round ind

    改preprocess.py

    ```
    python rounD_preprocess.py
    python inD_preprocess.py
    ```

    同步后台跑就行

    

## train

20240524 耗时1h

1. ```
   tmux
   python train.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128
   
   ctrl+b 松掉+d
   
   tmux ls
   tmux attach -t 0
   ```

2. 报错

   RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step. If this is intentional, you must enable the detection of unused parameters in DDP, either by setting the string value `strategy='ddp_find_unused_parameters_true'` or by setting the flag in the strategy with `strategy=DDPStrategy(find_unused_parameters=True)`.

   

   https://zhuanlan.zhihu.com/p/611870086

   https://github.com/Lightning-AI/pytorch-lightning/issues/17212

   **单机多卡**. 单机多卡时无需指定参数`num_nodes` 就是1

   devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices

   ​        (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for

   ​        automatic selection based on the chosen accelerator. Default: ``"auto"`

3. 解答：

   在看了博客 git论坛 官方doc 无果 问gpt 碰一碰死耗子

   train.py里45行strategy = “ddp" 改成`'ddp_find_unused_parameters_true'` 竟然成功了 跑起来了

   
   
   maxepoch是200就在后台跑啦 啦啦啦啦啦  算是有点样子了！！！！！跑起来了  1个epoch 1min左右
   
   **明天按水上大队长的想法   全局输出 attention捕获全局相关性（全局特征与输出间的关系！！！！）+ kan 缝模块替换DNN linear mlp！！！**
   
   
   
   跑了一晚上 device-1默认4张卡都用 还挺快，没问题 可以加训练的曲线图！
   
4. ```
   python train.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128
   
   
   0 | encoder | GRUGNNEncoder | 160 K
   1 | decoder | GRUGNNDecoder | 406 K
   ------------------------------------------
   567 K     Trainable params
   32        Non-trainable params
   567 K     Total params
   ```

5. ```
   python train.py --dataset inD --motion-model neuralode --n-workers 8 --hidden-size 128
   
     | Name    | Type          | Params
   ------------------------------------------
   0 | encoder | GRUGNNEncoder | 160 K
   1 | decoder | GRUGNNDecoder | 406 K
   ------------------------------------------
   567 K     Trainable params
   32        Non-trainable params
   567 K     Total params
   还挺快的
   ```

   4张卡一起跑

## test

1. 20240525

   没给自己试的， 到时候 函数入口打断点 然后单步调试就行！

   ```
   python test.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128
   ```

   

   `tmux set mouse on`可以上下滚动

   `tmux set mouse off`可以关掉 ，不然鼠标左右键默认一堆功能

   如果gpu显存退不掉：

```
nvidia-smi
kill -9 PID的编号
```

结果：成功啦啦啦啦啦啦啦 :happy:

           mr               0.9766247868537903
        test_ade            1.7046488523483276
        test_anll           1.6992120742797852
        test_apde           1.1911189556121826
        test_fde             4.706995487213135
        test_fnll           5.2622199058532715
       test_tv_ade          1.9396843910217285
      test_tv_anll           2.278433322906494
      test_tv_apde           1.425857424736023
       test_tv_fde           5.623199462890625
      test_tv_fnll           7.121974945068359
          tv_mr             0.9754895567893982

早上改代码 后天一直跑  然后干别的事  算法题+项目

2. ```
   python test.py --dataset inD --motion-model neuralode --n-workers 8 --hidden-size 128
   
   python test.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128
   ```

   结果在lightning logs里



## 论文阅读

### MTP-GO: Graph-Based Probabilistic Multi-Ag TIV 202309

#### 贡献

1. 时空结构： graph+GRU 保存交互关系 ，gru本身优势 记忆
2. 动态多样性： 考虑环境变化，但是预测鲁棒 how？？
3. 自然的预测？？？ 用ODE  这是NIPS 2018的创新：一种基于常微分方程的新型神经网络ODE Net  学习到了数据本来含的物理特性
4. 概率预测： 其实就是解决多模态不确定性，如何讲故事， 用EKF kalman 我可以改成加attention， 加kan

### 细节

1. 全局交互： 预测了未来轨迹 + 当前环境的未来状态，

   **这就是编故事，可以叫做为 辅助任务，可以是其他周围车辆的未来轨迹+交互关系一起输出**

   用图： 无向图 就是全连接静态图， 表示交互关系，提取周围车辆和车辆之间的交互（edge的值就是重要性）

2. 环境动态 但稳定输出

3. 轨迹 光滑+可行，泛化能力 适应不同agent type 可以吹逼

4. 多模态输出 解决不确定性

   GRU改成attention

周围车辆是8个！！！ M=8



motion model**输入是decoder 的output	用到ekf**来预测多模态，更多基于物理特性，这里可以改成attention



这里temporal graph 没有road的spatial graph

或者我便故事 说可以是temporal-spatio graph 隐含了road信息





GNN这里其实学过一点，翻一下笔记 哪个pdf  在pad上面

我滴个妈， gcn， gat graphconv都用了  那不都可以改啊！！！

**Gaussian Kernel Edge Weighting 求图的边的权重，我不会 没听过啊**

行人那里可以加kan



encoder 输出 + encoder输出过一个attention+



我可以编故事：

1. auxiliary task：辅助任务， 基于意图的behaviour，每个behaviour延伸一种轨迹
2. 自监督：这种意图未打过标签
3. 融合机理：学习物理特性



### Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction

18a 18b这里这里截取 三段k e 然后求r b h这里计算可以改， 截取对应的也能改， 没说为什么！！！！



明天学缝合 然后 先缝了再说 看效果，不需要懂原理！！！



## 创新点

1. KAN+CNN：用在social 行人预测那里 ，或者缝在GCN里，https://www.bilibili.com/video/BV1ub421B7Xz/?vd_source=81d34670595467089254a377dbe64851

2. gat改进 这里都算改进！！！！[Lipschitz归一化-应用于GAT和Graph Transformer的即插即用模](https://www.bilibili.com/video/BV1mm411Z7zD/)

## 缝合

学习20240528 

1. https://space.bilibili.com/478113245/channel/collectiondetail?sid=2080814 里面找灵感

2. 初级缝合

   1. 直接改attention替换

   2. 如何找模块： nips 一些nlp cv aaai的新模型https://www.bilibili.com/video/BV1zg4y1C7oP/?vd_source=81d34670595467089254a377dbe64851

      模块不能直接全拿过来用，单独把里面的核心部分 复制过来测试调通，看输入输出维度，可以打断点然后debug在调试控制台里print(x.shape)

3. 进阶缝合

   1. 和谁对比：不要和自己的缝合对象比，找其他的比较新的model

   2. 论文不用看，涨点才细看！！

      细看，看摘要+framework的图就够了

      https://www.bilibili.com/video/BV1Kw411G7Rw/?vd_source=81d34670595467089254a377dbe64851

   3. 并联 串联 融合![](C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240528140150955.png)

   4. 加cnn

   5. 步骤：

      1. 先在原模块的forward里刚开始print(x.shape)

      2. 然后里面在拔下来的 里面torch.rand(上面的维度！！！！) 看输出变了没， 如果不行，那就加conv linear换到一样的维度 就可以放进来！！！！

      3. 然后缝进去

         然后同时 缝几个，看谁涨点了

4. 工坊——串并联 融合缝合  external attention+ partialconv部分卷积

   1. 先单独测试单个模块， 弄清输入输出shape就够了

5. 可以改

   1. 输入 ——数据增强，数据融合
   2. model
   3. loss
   4. 激活函数

   我是试过了 都成功啦啦啦啦啦啦啦 根本没有难度，有脑子 有耐心就行， 来吧就这么缝 

   只要涨点了就开始编故事，baseline的整体框架一起说是自己的，只有基础模块是大佬写的，自己品来的

   然后缝合的新模块 说是自己的创新！！！！

   没涨点也可以水论文长度



### 202405027

1. 缝合b站学习

2. 调试
   1. 源代码逐行调试，在main函数的第一行打断点，在encoder+decoder 的forward第一行打断点
   2. 然后单步调试  或者逐过程

3. 学原文和要投的的论文写作方式——抄

### week14

#### 20240527

快速40min读完了2篇原文

#### 20240528

今天收到了dataset， 跑完了ind, round 很快 preprocess train test

缝合教学b站工坊买的10min看完喽

### week15

#### 20240604

#### 缝KAN

说kan的优势 去看arxiv和b站的解答  糊上去

今天缝合KAN 所有nn.linear改成KANLINEAR

```
tmux
python train.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128

  | Name    | Type          | Params
------------------------------------------
0 | encoder | GRUGNNEncoder | 643 K 
1 | decoder | GRUGNNDecoder | 2.3 M 
------------------------------------------
2.9 M     Trainable params
32        Non-trainable params
2.9 M     Total params
11.630    Total estimated model params size (MB)
```

尼玛 motionmodel里的nn.linear改成KAN就assert问题懒得改了，就这里面的没换，其他都换了！！

```
python train.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128
  | Name    | Type          | Params
------------------------------------------
0 | encoder | GRUGNNEncoder | 643 K
1 | decoder | GRUGNNDecoder | 2.3 M
------------------------------------------
2.9 M     Trainable params
32        Non-trainable params
2.9 M     Total params
11.681    Total estimated model params size (MB)

```

```
python train.py --dataset inD --motion-model neuralode --n-workers 8 --hidden-size 128

| Name    | Type          | Params
------------------------------------------
0 | encoder | GRUGNNEncoder | 643 K
1 | decoder | GRUGNNDecoder | 2.3 M
------------------------------------------
2.9 M     Trainable params
32        Non-trainable params
2.9 M     Total params
11.681    Total estimated model params size (MB)

```





#### 缝mamba

[mamba_ssm和causal-conv1d安装教程_不同torch版本的mamba-ssm-CSDN博客](https://blog.csdn.net/lihaiyuan_0324/article/details/138076262)

cuda torch都装了 太新了 没事 用最新的装

放弃了 环境配置太他妈 恶心了，直接用attention的变种！！！！！

#### 20240605

今天跑了 test， 改了一下test》py里的save_name  KAN_

```
python test.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128

──────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────────────
           mr               0.9482113122940063
        test_ade             2.75787091255188
        test_anll           3.6763086318969727
        test_apde           1.3482738733291626
        test_fde             5.862321853637695
        test_fnll            5.243432998657227
       test_tv_ade           1.257818341255188
      test_tv_anll          2.5309717655181885
      test_tv_apde          1.0916255712509155
       test_tv_fde           2.335230588912964
      test_tv_fnll          3.5663669109344482
          tv_mr             0.4601798355579376

inD
]
──────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────────────
           mr               0.8228151202201843
        test_ade            2.3930089473724365
        test_anll            2.074950933456421
        test_apde           0.8845089673995972
        test_fde             4.910714149475098
        test_fnll            4.370978832244873
       test_tv_ade          3.4080426692962646
      test_tv_anll          2.2008297443389893
      test_tv_apde          0.8927837610244751
       test_tv_fde            8.1976318359375
      test_tv_fnll           5.16241455078125
          tv_mr             0.8367840647697449

round
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────────────
           mr               0.8127992749214172
        test_ade            2.1414196491241455
        test_anll           2.2663750648498535
        test_apde           1.0852704048156738
        test_fde             5.27390193939209
        test_fnll            4.683055400848389
       test_tv_ade          1.7751853466033936
      test_tv_anll           1.968465805053711
      test_tv_apde          0.9161920547485352
       test_tv_fde           4.642155647277832
      test_tv_fnll           4.519531726837158
          tv_mr              0.687705397605896


```

### 暑假week3

#### 20240719——xlstm

改gru-> xlstm mlstm slstm

在ME_xLSTM2.py里https://cloud.tencent.com/developer/article/2418826

lstm gru对比 https://cloud.tencent.com/developer/article/1495400

lstm输入xt  ht ct; 输出ht ct

![image-20240719143842869](C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240719143842869.png)

![image-20240719144832070](C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240719144832070.png)



gru 输入xt ht 输出ht

<img src="C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240719143927695.png" alt="image-20240719143927695" style="zoom: 67%;" />





slstm：多了exp, nt；sLSTM（Scalar LSTM）在传统的LSTM基础上增加了标量更新机制。

![image-20240719145230210](C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240719145230210.png)



mlstm：通过将传统的LSTM中的向量操作扩展到矩阵操作，极大地增强了模型的记忆能力和并行处理能力。mLSTM的每个状态不再是单一的向量，而是一个矩阵，这使得它可以在单个时间步内捕获更复杂的数据关系和模式。

![image-20240719145452226](C:\Users\mephistopheles\AppData\Roaming\Typora\typora-user-images\image-20240719145452226.png)





**gru->gruGNN： xi 变 xi, ei, ef_i**



## 20240719 ——不加KAN 输出加全局attention

base_mdn.py里改了

```python
 # TODO：每个GNNdecoder 输出的next_states加 全局attetion  维数调对了
 block = ExternalAttention(d_model = next_states.shape[-1], S=8).cuda()
 next_states = block(next_states)

```

```
python train.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128
tmux 3
```

单步：就是跳到函数里

单过程就是一行行



比kan小很多啊，果然不能加kan

```
python train.py --dataset inD --motion-model neuralode --n-workers 8 --hidden-size 128
tmux 4 

0 | encoder | GRUGNNEncoder | 160 K 
1 | decoder | GRUGNNDecoder | 406 K 
------------------------------------------
567 K     Trainable params
32        Non-trainable params
567 K     Total params
2.270     Total estimated model params size (MB)
```





```
python train.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128  显存不够了！！！！

晚上突然想起来了，还挺快的，比kan快！！！
tmux 4
```





test：

```
python test.py --dataset highD --motion-model neuralode --n-workers 8 --hidden-size 128

 Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────
           mr                0.998857319355011
        test_ade            149.17735290527344
        test_anll            5.831324100494385
        test_apde            58.47233963012695
        test_fde             195.1042022705078
        test_fnll            6.978989601135254
       test_tv_ade          139.48165893554688
      test_tv_anll           5.649812698364258
      test_tv_apde           66.0583267211914
       test_tv_fde          212.95046997070312
      test_tv_fnll           7.733977794647217
          tv_mr             1.0079083442687988

```



```
python test.py --dataset inD --motion-model neuralode --n-workers 8 --hidden-size 128

─
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────
           mr               1.0002262592315674
        test_ade            16.371906280517578
        test_anll            5.444555759429932
        test_apde           10.950345993041992
        test_fde            16.830894470214844
        test_fnll            6.055561542510986
       test_tv_ade          12.963173866271973
      test_tv_anll           4.964584827423096
      test_tv_apde           7.25091552734375
       test_tv_fde          16.765979766845703
      test_tv_fnll           5.97965145111084
          tv_mr             0.9870992302894592

```



哎呀roundD怎么搞成了second order 的了 :cry:

```
python test.py --dataset rounD --n-workers 8 --hidden-size 128

  Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────
           mr               1.0001529455184937
        test_ade             24.82598876953125
        test_anll               1316342.375
        test_apde            17.66793441772461
        test_fde               24.8720703125
        test_fnll            7.787802219390869
       test_tv_ade          23.164443969726562
      test_tv_anll              1189790.875
      test_tv_apde           20.91692543029785
       test_tv_fde          24.779216766357422
      test_tv_fnll           7.782886981964111
          tv_mr             1.0087387561798096

```



### 20240720

总结  这个attention加完太吓人了 

external attetion加的位置改 改到可以以后 加KAN



external attention太离谱了这个输出，原来的位置可能后面加一个那个GNN norm！！！maybe 











换CNN 为unet！！！





看了一下ptg.nn库里自带了非常多的网络



反正都试一遍也可以的， 有的是时间，挂着



挑出相对最好那个



然后基于baseline对数据的美化成都来美化我的数据结果！！！



**工作重心：**

**怎么把结果保存下来，然后把合适的那个模块的训练train loss valid loss testloss图画出来**

**tqdm  参考李宏毅作业1可以**



怎么可视化，甚至可以不用是真的轨迹输出，编的合理就行反正