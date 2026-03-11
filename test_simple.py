
import tensorflow as tf
try:
    model = tf.keras.models.load_model('models/breast_cancer_model.h5', compile=False)
    print("SUCCESS")
    print(f"Shape: {model.input_shape}")
except Exception as e:
    print(f"FAILURE: {e}")
