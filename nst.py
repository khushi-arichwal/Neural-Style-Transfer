import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :]
    return img / 255.0

def mini_model(layer_names, model):
    outputs = [model.get_layer(name).output for name in layer_names]
    return Model([model.input], outputs)

def gram_matrix(tensor):
    t = tf.squeeze(tensor)
    fun = tf.reshape(t, [t.shape[2], t.shape[0]*t.shape[1]])
    result = tf.matmul(fun, fun, transpose_b=True)
    return result[tf.newaxis, :]

class Custom_Style_Model(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(Custom_Style_Model, self).__init__()
        self.vgg =  mini_model(style_layers + content_layers, vgg)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) 
                         for style_output in style_outputs]
        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}

def total_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = total_loss(outputs)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return loss

content_image_path = r"D:\data\github\Neural-Style-Transfer\dataset\content-images\main_building.jpeg"
style_image_path = r"D:\data\github\Neural-Style-Transfer\dataset\style-images\Oscar Florianus Bluemner - Old Canal Port.jpg"
content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = True
fine_tune_at = 100
for layer in vgg.layers[:fine_tune_at]:
    layer.trainable =  False
content_layers = ['block3_conv2', 'block4_conv2', 'block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

extractor = Custom_Style_Model(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

opt = tf.optimizers.Adam(learning_rate=0.01)

style_weight=1e-2
content_weight=1e5

target_image = tf.Variable(content_image)

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        loss = train_step(target_image)
        print(".", end='')
    plt.figure(figsize=(10,10))
    plt.imshow(np.squeeze(target_image.read_value(), 0))
    plt.title("Train step: {}".format(step))
    plt.show()  
    print("Total loss: ", loss.numpy())
