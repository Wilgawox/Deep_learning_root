import tensorflow.keras.backend as K
import tensorflow as tf


def apply_threshold(y_pred):
    return K.round(y_pred)

def tp_custom(y_true, y_pred):
    return K.sum(K.round(y_true * apply_threshold(y_pred)))

def tn_custom(y_true, y_pred):
    return K.sum(K.cast(K.equal(K.round(y_true + apply_threshold(y_pred)), 0), K.floatx()))

def fp_custom(y_true, y_pred):
    return K.sum(K.cast(K.equal(K.round(apply_threshold(y_pred)) - y_true, 1), K.floatx()))

def fn_custom(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true - K.round(apply_threshold(y_pred)), 1), K.floatx()))

def pos_custom(y_true, y_pred):
    return K.sum(K.round(y_true))

def neg_custom(y_true, y_pred):
    return K.sum(K.round(1-y_true))

def precision_custom(y_true, y_pred):
    return tp_custom(y_true, y_pred) / (tp_custom(y_true, y_pred) + fp_custom(y_true, y_pred))

def recall_custom(y_true, y_pred):
    return tp_custom(y_true, y_pred) / (tp_custom(y_true, y_pred) + fn_custom(y_true, y_pred))

def bce_custom(y_true, y_pred, sample_weight=[1,1]):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    print(y_true)
#    return bce(y_true, y_pred)
    output=tf.convert_to_tensor(y_pred)
    epsilon_=tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    target=tf.convert_to_tensor(y_true)
    bce = K.cast(sample_weight[1],K.floatx()) * K.cast(tf.math.log(output + K.epsilon()),K.floatx())
    bce += K.cast(sample_weight[0],K.floatx()) * K.cast(1 - target,K.floatx()) * K.cast(tf.math.log(1 - output + K.epsilon()),K.floatx())
    return -bce

