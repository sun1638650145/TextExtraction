import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer


def decode_prediction(y, dataframe, max_length, tokenizer):
    """Decode prediction data into human-readable texts.

    Args:
        y: Tensor,
            tensor predicted by the model.
        dataframe: pd.core.DataFrame,
            the raw dataframe.
        max_length: int,
            maximum length of text.
        tokenizer: tokenizers.ByteLevelBPETokenizer,
            a tokenizer, it will be used to decode.

    Returns:
        List of human-readable texts.
    """
    preds_start = np.zeros((dataframe.shape[0], max_length))
    preds_end = np.zeros((dataframe.shape[0], max_length))
    preds_start += y[0]
    preds_end += y[1]

    ans = []
    for i in range(dataframe.shape[0]):
        start = np.argmax(preds_start[i, ])
        end = np.argmax(preds_end[i, ])
        if start > end:
            selected_text = dataframe.loc[i, 'text']
        else:
            text = ' ' + ' '.join(dataframe.loc[i, 'text'].split())
            encode = tokenizer.encode(text)
            selected_text = tokenizer.decode(encode.ids[start - 1: end])
        print(i, selected_text)
        ans.append(selected_text)

    return ans


def _get_tokenizer(tokenizer_cache_dir):
    """Get tokenizer.

    Args:
        tokenizer_cache_dir: str,
            configuration data of the pre-trained tokenizer.

    Returns:
        tokenizers.ByteLevelBPETokenizer
    """
    if not os.path.exists(tokenizer_cache_dir):
        os.makedirs(tokenizer_cache_dir)

    robert_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='roberta-base')
    robert_tokenizer.save_pretrained(tokenizer_cache_dir)

    tokenizer = ByteLevelBPETokenizer(vocab=tokenizer_cache_dir+'vocab.json',
                                      merges=tokenizer_cache_dir+'merges.txt',
                                      add_prefix_space=True,
                                      lowercase=True)

    return tokenizer


def _load_dataframe(dataset_path):
    """Load the dataframe.

    Args:
        dataset_path: str or path,
            the path of dataset.

    Returns:
        Pandas.core.DataFrame
    """
    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe.fillna('')

    return dataframe


def select_device(device='GPU'):
    """Select the device on which the task runs,
     and initialize the strategy instance.

    Args:
        device: {'GPU', 'TPU'}, default='GPU',
            The device running the task.

    Returns:
        tf.distribute.strategy
    """
    if device == 'TPU':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    return strategy