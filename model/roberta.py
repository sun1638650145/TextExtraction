import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate, Dense, Dropout, Flatten,  Input
from tensorflow.keras.models import Model
from transformers import TFRobertaModel, RobertaConfig


def question_answering_model(max_length):
    """Instantiates the Question Answering Model.

    Args:
        max_length: int, the number of sequence.

    Returns:
        A Keras model instance.
    """
    input_ids = Input(shape=(max_length,), name='Input-Ids', dtype=tf.int32)
    attention_mask = Input(shape=(max_length,), name='Attention-Mask', dtype=tf.int32)
    token_type_ids = Input(shape=(max_length,), name='Token-Type-Ids', dtype=tf.int32)

    # Roberta backbone
    roberta_backbone = TFRobertaModel.from_pretrained(pretrained_model_name_or_path='roberta-base',
                                                      config=RobertaConfig.from_pretrained('roberta-base'))
    sequence_output, pooled_output = roberta_backbone(inputs=input_ids,
                                                      attention_mask=attention_mask,
                                                      token_type_ids=token_type_ids)

    # output layer
    end_tokens_layer = Dropout(rate=0.1, name='End-Dropout')(sequence_output)
    end_dense_layer = Dense(units=1, name='End-Dense')(end_tokens_layer)
    end_tokens_layer = Flatten(name='End-Flatten')(end_dense_layer)
    end_tokens_layer = Activation(activation='softmax', name='end_tokens')(end_tokens_layer)

    start_tokens_layer = Concatenate(name='Start-Concatenate')([end_dense_layer, sequence_output])
    start_tokens_layer = Dropout(rate=0.1, name='Start-Dropout')(start_tokens_layer)
    start_tokens_layer = Dense(units=1, name='Start-Dense')(start_tokens_layer)
    start_tokens_layer = Flatten(name='Start-Flatten')(start_tokens_layer)
    start_tokens_layer = Activation(activation='softmax', name='start_tokens')(start_tokens_layer)

    model = Model(inputs=[input_ids, attention_mask, token_type_ids],
                  outputs=[start_tokens_layer, end_tokens_layer],
                  name='question_answering_model')

    return model