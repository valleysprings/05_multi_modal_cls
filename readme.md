# 当代人工智能大作业——多模态情感分析



##  注意事项

-   全程在服务器端部署并测试。
-   进入项目目录，将给定的dataset(MVSA-Single)放入data目录中。
-   按照`requirement.txt`安装所需要的依赖（注：clip可能需要build from source，从repo下即可）
-   本实验前半部分需要生成给定的模型embedding（训练与测试），因此首先需要执行以下代码：

```Bash
# 生成clip对应文本以及图像embedding
python feats/clip_extract_feats.py 

# 生成roberta文本embedding
python feats/roberta_extract_feats.py 

# 生成resnet图像embedding
python feats/resnet_extract_feats.py 
```

-   然后进行head训练，执行以下代码：

```Bash
python modeling/linear-probe/train_head.py 
训练超参数可选择：

- 视觉embedding（`vtype`），默认为clip
- 文本embedding（`ttype`），默认为clip
- 批量大小（`bs`），默认为512
- 学习率（`lr`），默认为1e-4
- 训练周期（`ep`），默认为100
- 分类头大小（`D`），默认为256
- 开启哪个通道（`ablation`），默认为0（开启视觉与文本双通道）

具体设置请见`train_head.py`
```

-   最后进行推理，执行以下代码：

```Bash
python modeling/linear-probe/inference_head.py
```

-   在此项目添加了对roberta与resnet的fine-tuning部分，执行以下代码进行复现：

```Bash
python modeling/fine-tuning/finetuning.py
```

## 代码结构

```bash
05_multi_modal_cls
├─data											#把实验数据集放进去
│ 
├─feats											#特征提取
│      clip_extract_feats.py					#处理clip特征				
│      resnet_extract_feats.py					#处理resnet特征
│      roberta_extract_feats.py					#处理roberta特征
│
├─modeling										#训练与预测
│  ├─fine-tuning								#调优特征
│  │      finetuning.py							#调优部分
│  │      model_fine_tuning.py					#模型定义
│  │
│  └─linear-probe										#静态特征
│          inference_head.py					#推理部分
│          model_head.py						#模型定义
│          train_head.py						#训练部分
│
└─saved											#暂存区
    ├─saved_feats								#特征暂存
    └─saved_models								#模型暂存
```

## 效果

### 验证集结果

使用如下三种模型进行验证：

1.  clip：ViT-B/32、ViT-L/14
2.  roberta：roberta-base
3.  resnet：resnet50

fine-tuning的batch size设置为64，训练分类头batch size设置为512。

| 视觉encoder | 文本encoder | encoder是否进行fine-tune | 分类头模态通道大小D | 学习率lr | 验证集最佳准确率 |
| ----------- | ----------- | ------------------------ | ------------------- | -------- | ---------------- |
| clip        | clip        | 否                       | 128                 | 2e-5     | 0.755            |
| clip        | clip        | 否                       | 128                 | 1e-4     | 0.775            |
| clip        | clip        | 否                       | 256                 | 2e-5     | 0.775            |
| clip        | clip        | 否                       | 256                 | 1e-4     | 0.787            |
| resnet50    | roberta     | 否                       | 128                 | 2e-5     | 0.682            |
| resnet50    | roberta     | 否                       | 128                 | 1e-4     | 0.698            |
| resnet50    | roberta     | 否                       | 256                 | 2e-5     | 0.688            |
| resnet50    | roberta     | 否                       | 256                 | 1e-4     | 0.718            |
| resnet50    | roberta     | 是                       | 256                 | 2e-5     | 0.723            |
| resnet50    | roberta     | 是                       | 256                 | 1e-4     |                  |

使用ViT-L/14进一步进行测试，超参数沿用之前实验：

| 分类头模态通道大小D | 学习率lr | 验证集最佳准确率 |
| ------------------- | -------- | ---------------- |
| 128                 | 2e-5     | 0.750            |
| 128                 | 1e-4     | 0.785            |
| 256                 | 2e-5     | 0.767            |
| 256                 | 1e-4     | 0.777            |

发现效果更差了？可能是这种融合方式没有特别好利用CLIP所标的的信息。

我想看一下批量大小对训练的影响，批量大小可能太大了（512），降至32重新测试。

| 分类头模态通道大小D | 学习率lr | 验证集最佳准确率 |
| ------------------- | -------- | ---------------- |
| 128                 | 2e-5     | 0.777            |
| 128                 | 1e-4     | 0.787            |
| 256                 | 2e-5     | 0.780            |
| 256                 | 1e-4     | 0.775            |

好像训练起来效果好了一点，但这个提升真的有点微不足道（尤其是使用了fp16且验证集大小只有400张图像时，其实验证集波动挺大的）。

### 消融测试

学习率使用之前测试最好的学习率，超参设置与前一节保持一致。

| 视觉encoder | 文本encoder | encoder是否进行fine-tune | 验证集准确率 |
| ----------- | ----------- | ------------------------ | ------------ |
| clip        | -           | 否                       | 0.762        |
| -           | clip        | 否                       | 0.767        |
| resnet50    | -           | 否                       | 0.693        |
| -           | roberta     | 否                       | 0.708*       |
| resnet50    | -           | 是                       | 0.738        |
| -           | roberta     | 是                       | 0.675*       |

*：模型有可能无法收敛。



## 参考实现

https://github.com/cleopatra-itn/fair_multimodal_sentiment



## 文献

1.  Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." *arXiv preprint arXiv:2010.11929* (2020).
2.  Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." *arXiv preprint arXiv:1907.11692* (2019).
3.  He, Pengcheng, et al. "Deberta: Decoding-enhanced bert with disentangled attention." *arXiv preprint arXiv:2006.03654* (2020).
4.  He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.
5.  Radford, Alec, et al. "Learning transferable visual models from natural language supervision." *International Conference on Machine Learning*. PMLR, 2021.
6.  Dou, Zi-Yi, et al. "An empirical study of training end-to-end vision-and-language transformers." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2022.
7.  Cheema, Gullal S., et al. "A fair and comprehensive comparison of multimodal tweet sentiment analysis methods." *Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding*. 2021.
