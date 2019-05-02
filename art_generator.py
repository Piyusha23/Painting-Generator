import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input of VGG16
    image = image - np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    return image

def generate_noise_image(content_image, noise_ratio = 0.6):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, 300, 400, 3)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(tf.transpose(a_C),[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(tf.transpose(a_G),[n_H*n_W,n_C])

    J_content = (1/(4*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))))

    return J_content

def compute_style_cost(model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = alpha*J_content + beta*J_style

    return J

tf.reset_default_graph()

sess = tf.InteractiveSession()

content_image = scipy.misc.imread("images/CL-Building-2.png")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/starry_night.png")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_1']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))

J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha = 30, beta = 40)

optimizer = tf.train.AdamOptimizer(0.5)

train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 300):

    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("output/" + str(i) + ".png", generated_image)

    save_image('output/generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)
