import torchvision.models as models
import onnx
import torch.nn as nn
import torch
import onnx_tf
import tensorflow as tf
import tensorflow_probability as tfp


model = models.mobilenet_v2()
in_features = model._modules['classifier'][-1].in_features
model._modules['classifier'][-1] = nn.Linear(in_features, 36, bias=True)
model.eval()

input_shape=(1,3,128,128)
dummy_input=torch.randn(input_shape)
onnx_model_path='mobilenet_v2.onnx'
torch.onnx.export(model,dummy_input,onnx_model_path,verbose=False)

onnx_model= onnx.load(onnx_model_path)
tf_model_path='mobilenet_v2.pb'
if_rep=onnx_tf.backend.prepare(onnx_model)
if_rep.export_graph(tf_model_path)

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

with open('mobilenet_v2.tflite','wb') as f :
    f.write(tflite_model)