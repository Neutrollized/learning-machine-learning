# this doesn't work

import tensorflow as tf
import tensorflow_hub as hub

#model = tf.keras.models.load_model("cats_and_dogs.h5")
model = tf.keras.models.load_model("cats_and_dogs.h5", custom_objects={'KerasLayer':hub.KerasLayer})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_saved_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8	# or tf.uint8
converter.inference_output_type = tf.int8	# or tf.uint8

tflite_quantized_model = converter.convert()

with open('your_qualtized_model.tflite', 'wb') as f:
  f.write(tflite_quantized_model)
