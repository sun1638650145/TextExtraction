# TextExtraction

一个关于文本提取的AutoML，你可以使用三四行代码就可以进行文本提取，

阅读其他语言版本：[简体中文](https://github.com/sun1638650145/TextExtraction/blob/main/README.md)、[English](https://github.com/sun1638650145/TextExtraction/blob/main/README-en.md)

## 例子

### pipeline

这是一个非常简短的例子（入门推荐）

```python
from TextExtraction import TextExtractionPipeline
pipeline = TextExtractionPipeline(train_dataset_path='../Sentiment_Extraction103/train.csv')
pipeline.run()
```

例子中使用的数据集在这里[点击](https://data.yanxishe.com/Sentiment_Extraction103.zip)

### custom

如果你希望更高的准确率，可以使用自定义模式（针对有经验的开发者）

1. 你可以使用model和tools下的API构建你自己的模型
2. 有问题欢迎和作者交流，联系方式qq:1638650145，邮箱:s1638650145@gmail.com

## 性能

1. 使用Jaccard系数评估，Jaccard系数在0.69-0.70之间
2. 使用tf.data.Datasets读入数据集并使用默认参数，在Nvidia Tesla P100上的参考速度是370ms/step， 在Google TPU上参考速度是101ms/step
3. 支持使用TPU运行

## 如果你想

1. 如果你想改进代码，请使用PEP8标准，否则一定无法通过
2. 如果你想使用其他的模型比如Bert、Albert，请与作者交流，联系方式在上面
3. 如果你想star和fork，那就不用想了直接做就行了，你的想法非常明智，最后，非常感谢你的支持