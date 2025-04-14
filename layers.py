# custom L1 Distance layer module

# import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer 

# custom L1Distance Layer from jupyter
class L1Dist(Layer):
    # init metod inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    #magic happens here - similrity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    