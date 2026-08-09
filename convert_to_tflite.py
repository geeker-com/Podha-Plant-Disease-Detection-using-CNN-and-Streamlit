import tensorflow as tf

MODEL_IN = "resnet50_plant_disease_final_96.h5"
MODEL_OUT = "resnet50_plant_disease_final_96.tflite"

model = tf.keras.models.load_model(MODEL_IN, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(MODEL_OUT, "wb") as f:
    f.write(tflite_model)

print(f"Saved {MODEL_OUT}")
