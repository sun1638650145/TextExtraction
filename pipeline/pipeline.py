import logging

import pandas as pd
from tensorflow.keras.backend import clear_session
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from TextExtraction.tools import decode_prediction
from TextExtraction.tools import create_tf_dataset
from TextExtraction.tools import preprocessing_dataframe
from TextExtraction.tools import select_device
from TextExtraction.model import question_answering_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='TextExtraction')


class TextExtractionPipeline(object):
    """TextExtraction pipeline.
    You only need to configure a few parameters and specify the path of the dataset,
     the pipeline can fit automatically.

    Attributes:
        train_dataset_path: str,
            the path of train dataset, the file must be a csv file.
        test_dataset_path: str, default=None
            the path of test dataset, the file must be a csv file.
        model_savedpath: str or path, default='./roberta-weights.h5',
            h5 file for saving the model's weights.
        tokenizer_cache_dir: str, default='./roberta/',
            configuration data of the pre-trained tokenizer.
        save_inference: str, default=None,
            save the inference result as a csv file.
        max_length: int, default=96,
            maximum length of text.
        batch_size: int, default=32.
            number of samples per gradient update.
        learning_rate: float, default=4e-5,
            the optimization range of the optimizer.
        epochs: int, default=5,
            number of epochs to train the model.
        device: {'GPU', 'TPU'}, default='GPU',
            the physical device where the pipeline will run.
        text_column: str, default='text',
            raw text column in the dataset.
        selected_text_column: str, default='selected_text',
            answer text column in the dataset.
        sentiment_column: str, default='sentiment',
            sentiment information of the text in the dataset.
        inference_status: bool, default=False,
            if true the model will be trained, else the model will be trained and inferred.
        tokenizer: tokenizers.ByteLevelBPETokenizer,
            a text tokenizer.
    """
    def __init__(self,
                 train_dataset_path,
                 test_dataset_path=None,
                 model_savedpath='./roberta-weights.h5',
                 tokenizer_cache_dir='./roberta/',
                 save_inference=None,
                 max_length=96,
                 batch_size=32,
                 learning_rate=4e-5,
                 epochs=5,
                 device='GPU',
                 text_column='text',
                 selected_text_column='selected_text',
                 sentiment_column='sentiment',
                 inference_status=False):
        super(TextExtractionPipeline, self).__init__()
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model_savedpath = model_savedpath
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.save_inference = save_inference

        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device

        self.text_column = text_column
        self.selected_text_column = selected_text_column
        self.sentiment_column = sentiment_column

        self.inference_status = inference_status

        self.tokenizer = None

    def run(self):
        """Run pipeline."""
        clear_session()

        if self.inference_status is True and self.test_dataset_path is not None:
            train_tf_dataset, test_tf_dataset, test_dataframe, self.tokenizer = self._preprocessing()
        else:
            train_tf_dataset = self._preprocessing()
        logger.info('Dataset preprocessing is complete, training will start soon ...')

        # Instantiates model
        strategy = select_device(self.device)
        with strategy.scope():
            model = question_answering_model(self.max_length)
            model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                          loss=CategoricalCrossentropy())
        model.summary()
        logger.info('Model instantiation completed.')

        # Fit and save model
        model.fit(x=train_tf_dataset,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)
        model.save_weights(filepath=self.model_savedpath)
        logger.info('The model training is complete, and the save is complete.')

        # Model inference
        if self.inference_status is True and self.test_dataset_path is not None:
            y_pred = model.predict(x=test_tf_dataset, verbose=1)
            ans = decode_prediction(y_pred, test_dataframe, self.max_length, self.tokenizer)
            if self.save_inference is not None:
                dataframe = pd.DataFrame(ans)
                dataframe.to_csv(self.save_inference, index=True, header=False, encoding='utf-8')
        logger.info('Pipeline finished.')

    def _preprocessing(self):
        """Preprocessing dataset.

        Returns:
           Create a tf.data.Dataset for training.
        """
        if self.inference_status is True and self.test_dataset_path is not None:
            (x, y), train_dataframe, tokenizer = preprocessing_dataframe(self.train_dataset_path,
                                                                         self.max_length,
                                                                         self.tokenizer_cache_dir,
                                                                         self.text_column,
                                                                         self.selected_text_column,
                                                                         self.sentiment_column,
                                                                         task_type='train')
            (x_test), test_dataframe, tokenizer = preprocessing_dataframe(self.test_dataset_path,
                                                                          self.max_length,
                                                                          self.tokenizer_cache_dir,
                                                                          self.text_column,
                                                                          self.selected_text_column,
                                                                          self.sentiment_column,
                                                                          task_type='test')

            train_tf_dataset = create_tf_dataset(x, y, batch_size=self.batch_size, task_type='train')
            test_tf_dataset = create_tf_dataset(x_test, batch_size=self.batch_size, task_type='test')

            return train_tf_dataset, test_tf_dataset, test_dataframe, tokenizer
        else:
            (x, y), train_dataframe, tokenizer = preprocessing_dataframe(self.train_dataset_path,
                                                                         self.max_length,
                                                                         self.tokenizer_cache_dir,
                                                                         self.text_column,
                                                                         self.selected_text_column,
                                                                         self.sentiment_column,
                                                                         task_type='train')

            train_tf_dataset = create_tf_dataset(x, y, batch_size=self.batch_size, task_type='train')

            return train_tf_dataset