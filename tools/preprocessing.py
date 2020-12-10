import json

import numpy as np
import tensorflow as tf

from TextExtraction.tools.utils import _get_tokenizer
from TextExtraction.tools.utils import _load_dataframe


def _tokenization_dataframe(dataframe,
                            tokenizer,
                            max_length,
                            sentiment_id,
                            text_column,
                            selected_text_column,
                            sentiment_column,
                            task_type):
    """Tokenization dataframe.

    Args:
        dataframe: pd.core.DataFrame,
            the raw dataframe.
        tokenizer: tokenizers.ByteLevelBPETokenizer,
            a text tokenizer.
        max_length: int,
            maximum length of text.
        sentiment_id:
        text_column: str, default='text',
            raw text column in the dataset.
        selected_text_column: str, default='selected_text',
            answer text column in the dataset.
        sentiment_column: str, default='sentiment',
            sentiment information of the text in the dataset.
        task_type: str, default='train',
            task type.

    Returns:
        Input data and output_token.
    """
    num_of_sample = dataframe.shape[0]

    input_ids = np.ones(shape=[num_of_sample, max_length], dtype='int32')
    attention_mask = np.zeros(shape=[num_of_sample, max_length], dtype='int32')
    token_type_ids = np.zeros(shape=[num_of_sample, max_length], dtype='int32')

    start_tokens = np.zeros(shape=[num_of_sample, max_length], dtype='int32')
    end_tokens = np.zeros(shape=[num_of_sample, max_length], dtype='int32')

    for index in range(num_of_sample):
        text = ' ' + ' '.join(dataframe.loc[index, text_column].split())
        encoded_text = tokenizer.encode(text)
        s_token = sentiment_id[dataframe.loc[index, sentiment_column]]
        # Use <s>A</s></s>B</s> to mark sentence pairs.
        input_ids[index, : len(encoded_text.ids) + 5] = [0] + encoded_text.ids + [2, 2] + [s_token] + [2]
        # Add attention.
        attention_mask[index, :len(encoded_text.ids) + 5] = 1

        if task_type == 'train':
            selected_text = ' '.join(dataframe.loc[index, selected_text_column].split())
            idx = text.find(selected_text)
            # Mark the position of the label character.
            arr_text = np.zeros([len(text)])
            arr_text[idx: idx + len(selected_text)] = 1
            if text[idx - 1] == ' ':
                arr_text[idx - 1] = 1
            # Record the position of each word.
            offsets = []
            idx = 0
            for token in encoded_text.ids:
                word = tokenizer.decode([token])
                offsets.append((idx, idx + len(word)))
                idx += len(word)
            # Mark the position of the label word.
            tokens = []
            for i, (a, b) in enumerate(offsets):
                if np.sum(arr_text[a: b]) > 0:
                    tokens.append(i)
            # Locate the tag word (the tag word exists in the original sentence).
            if len(tokens) > 0:
                start_tokens[index, tokens[0] + 1] = 1
                end_tokens[index, tokens[-1] + 1] = 1

    x = (input_ids, attention_mask, token_type_ids)
    y = (start_tokens, end_tokens)

    if task_type == 'train':
        return x, y
    else:
        return x


def preprocessing_dataframe(dataset_path,
                            max_length,
                            tokenizer_cache_dir='./roberta/',
                            text_column='text',
                            selected_text_column='selected_text',
                            sentiment_column='sentiment',
                            task_type='train'):
    """Preprocess the input dataframe, use the tokenizer to tag data.

    Args:
        dataset_path: str,
            the path of dataset, the file must be a csv file.
        max_length: int,
            maximum length of text.
        tokenizer_cache_dir: str, default='./roberta/',
            configuration data of the pre-trained tokenizer.
        text_column: str, default='text',
            raw text column in the dataset.
        selected_text_column: str, default='selected_text',
            answer text column in the dataset.
        sentiment_column: str, default='sentiment',
            sentiment information of the text in the dataset.
        task_type: str, default='train',
            task type.

    Returns:
        Data after using tokenizer, raw dataframe and tokenizer.
    """
    tokenizer = _get_tokenizer(tokenizer_cache_dir)

    dataframe = _load_dataframe(dataset_path)

    sentiment = dataframe[sentiment_column].unique()
    sentiment_id = dict()
    _json_file = open(tokenizer_cache_dir+'vocab.json', mode='r')
    _vocab = json.load(_json_file)
    for value in sentiment:
        sentiment_id.update({value: _vocab['Ġ' + value]})

    return _tokenization_dataframe(dataframe,
                                   tokenizer,
                                   max_length,
                                   sentiment_id,
                                   text_column,
                                   selected_text_column,
                                   sentiment_column,
                                   task_type), dataframe, tokenizer


def create_tf_dataset(x,
                      y=None,
                      batch_size=32,
                      task_type='train'):
    """Create a tf.data.Dataset for training or inference.

    Args:
        x: Array-like,
            tuple of input data list.
        y: Array-like, default=None,
            if inference status is False, y is not None.
        batch_size: Int, default=32,
            number of samples per gradient update.
        task_type: str, default='train',
            task type.

    Returns:
        tf.data.Dataset
    """
    if task_type is 'train':
        input_ids, attention_mask, token_type_ids = x
        start_tokens, end_tokens = y
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids,
                                                       'attention_mask': attention_mask,
                                                       'token_type_ids': token_type_ids},
                                                      {'start_tokens': start_tokens,
                                                       'end_tokens': end_tokens}))
    else:
        input_ids, attention_mask, token_type_ids = x
        dataset = tf.data.Dataset.from_tensor_slices({'input_ids': input_ids,
                                                      'attention_mask': attention_mask,
                                                      'token_type_ids': token_type_ids})

    dataset = (
        dataset
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset