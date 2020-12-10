# TextExtraction

An AutoML about text extraction, you can use three or four lines of code to extract text.

Read in other languages：[简体中文](https://github.com/sun1638650145/TextExtraction/blob/main/README.md)、[English](https://github.com/sun1638650145/TextExtraction/blob/main/README-en.md)

## example

### pipeline

This is a very simple example.(Suitable for beginners)

```python
from TextExtraction import TextExtractionPipeline
pipeline = TextExtractionPipeline(train_dataset_path='../Sentiment_Extraction103/train.csv')
pipeline.run()
```

The dataset used in the example, click[here](https://data.yanxishe.com/Sentiment_Extraction103.zip)

### custom

If you want higher accuracy, you can use custom mode (Suitable for experts)

1. You can use the model API and tools API to build your own model.
2. If you have any questions, welcome to communicate with the author and contact information qq:1638650145, email:[s1638650145@gmail.com](mailto:s1638650145@gmail.com), and issue.

## Performance

1. Use Jaccard coefficient to evaluate, Jaccard coefficient is between 0.69-0.70.
2. Use tf.data.Datasets to load the dataset and use the default parameters. The reference speed on Nvidia Tesla P100 is 370ms/step, and the reference speed on Google TPU is 101ms/step.
3. Support running with TPU.

## If you want to do something

1. If you want to code, please use the PEP8, otherwise it must not pass.
2. If you want to use other models such as Bert and Albert, please communicate with the author. The contact information is above.
3. If you want to star and fork, just do it. Your idea is very wise. Finally, thanks.