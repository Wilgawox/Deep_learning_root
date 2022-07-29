import tensorflow.keras.backend as K
import tensorflow as tf


def apply_threshold(y_pred):
    '''Apply a threshold of 0.5 on y_pred'''
    return K.round(y_pred)

def tp_custom(y_true, y_pred):
    # Metric that calculate the numbers of true positives
    return K.sum(K.round(y_true * apply_threshold(y_pred)))

def tn_custom(y_true, y_pred):
    # Metric that calculate the numbers of true negatives
    return K.sum(K.cast(K.equal(K.round(y_true + apply_threshold(y_pred)), 0), K.floatx()))

def fp_custom(y_true, y_pred):
    # Metric that calculate the numbers of false positives
    return K.sum(K.cast(K.equal(K.round(apply_threshold(y_pred)) - y_true, 1), K.floatx()))

def fn_custom(y_true, y_pred):
    # Metric that calculate the numbers of false negatives
    return K.sum(K.cast(K.equal(y_true - K.round(apply_threshold(y_pred)), 1), K.floatx()))

def pos_custom(y_true, y_pred):
    #Metric that calculate the number of positive values
    return K.sum(K.round(y_true))

def neg_custom(y_true, y_pred):
    #Metric that calculate the number of positive values
    return K.sum(K.round(1-y_true))

def precision_custom(y_true, y_pred):
    #Metric that calculate the precision of a model (TP/(TP+FP))
    return tp_custom(y_true, y_pred) / (tp_custom(y_true, y_pred) + fp_custom(y_true, y_pred))

def recall_custom(y_true, y_pred):
    #Metric that calculate the recall of a model (TP/(TP+FN))
    return tp_custom(y_true, y_pred) / (tp_custom(y_true, y_pred) + fn_custom(y_true, y_pred))

def bce_custom(y_true, y_pred, sample_weight=[1,1]):
    # Custom loss function used to get the binary crossentropy
    # Use is not recommended, please try using keras.losses.BinaryCrossentropy
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    output=tf.convert_to_tensor(y_pred)
    epsilon_=tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    target=tf.convert_to_tensor(y_true)
    bce = K.cast(sample_weight[1],K.floatx()) * K.cast(tf.math.log(output + K.epsilon()),K.floatx())
    bce += K.cast(sample_weight[0],K.floatx()) * K.cast(1 - target,K.floatx()) * K.cast(tf.math.log(1 - output + K.epsilon()),K.floatx())
    return -bce


def focal_loss(target, output, gamma=50):
    # Definition of focal loss
    print(target, output)
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)


def weighted_categorical_crossentropy(weights):
    # Definition of weighted categorical crossentropy
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce


def f1_loss(y_true, y_pred):
    # Definition of f1 loss
    tp = K.sum(K.cast(y_true,K.floatx())*K.cast(y_pred,K.floatx()), axis=0)
    fp = K.sum(K.cast((1-y_true),K.floatx())*K.cast(y_pred,K.floatx()), axis=0)
    fn = K.sum(K.cast(y_true,K.floatx())*K.cast((1-y_pred),K.floatx()), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)