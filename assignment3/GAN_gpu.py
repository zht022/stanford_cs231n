from __future__ import print_function, division

# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')


import tensorflow as tf
import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

#%matplotlib inline
#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions


'''
def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn) ## http://blog.csdn.net/wizardforcel/article/details/54232788
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return
'''

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

answers = np.load('gan-checks-tf.npz')


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)


def leaky_relu(x, alpha=0.01):
    # TODO: implement leaky ReLU
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    # TODO: sample and return noise
    return tf.random_uniform(shape=[batch_size, dim], minval=-1, maxval=1)


def discriminator(x):
    with tf.variable_scope('discriminator'):
        # TODO: implement architecture
        input_layer = tf.reshape(x, [tf.shape(x)[0], 28, 28, 1])

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=4, strides=2, padding='valid')
        lrelu1 = leaky_relu(conv1, 0.01)

        conv2 = tf.layers.conv2d(inputs=lrelu1, filters=128, kernel_size=4, strides=2, padding='valid')
        lrelu2 = leaky_relu(conv2, 0.01)
        bn2 = tf.layers.batch_normalization(inputs=lrelu2, axis=3, training=True)
        bn2_flatten = tf.reshape(bn2, [tf.shape(bn2)[0], 5 * 5 * 128])

        fc3 = tf.layers.dense(inputs=bn2_flatten, units=1024, use_bias=True)
        lrelu3 = leaky_relu(fc3, 0.01)

        logits = tf.layers.dense(inputs=lrelu3, units=1, use_bias=True)
        return logits

def generator(z):
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        fc1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.relu, use_bias=True)
        bn1 = tf.layers.batch_normalization(inputs=fc1, axis=1, training=True)

        fc2 = tf.layers.dense(inputs=bn1, units=7 * 7 * 128, activation=tf.nn.relu, use_bias=True)
        bn2 = tf.layers.batch_normalization(inputs=fc2, axis=1, training=True)

        bn2_resize = tf.reshape(bn2, [tf.shape(bn2)[0], 7, 7, 128])

        conv_T3 = tf.layers.conv2d_transpose(inputs=bn2_resize, filters=64, kernel_size=4, strides=2,
                                             padding='same', activation=tf.nn.relu, use_bias=True)
        bn3 = tf.layers.batch_normalization(inputs=conv_T3, axis=3, training=True)

        conv_T4 = tf.layers.conv2d_transpose(inputs=bn3, filters=1, kernel_size=4, strides=2,
                                             padding='same', activation=tf.nn.tanh, use_bias=True)
        img = tf.reshape(conv_T4, [tf.shape(conv_T4)[0], 784])
        return img


def wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample):
    # TODO: compute D_loss and G_loss
    ## according to line 9 and line 12, we firstly need to calculate means then optimize
    ## In this notebook, we always take n_{critics}/k steps as 1
    D_loss = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)
    G_loss = tf.reduce_mean(logits_fake)

    # lambda from the paper
    lam = 10

    # random sample of batch_size (tf.random_uniform)
    # sampling real data and latent data is contained in function "run_a_gan"
    eps = tf.random_uniform([])  ## line 4: sample a random number from U[0,1]
    x_hat = eps * x + (1 - eps) * G_sample  ## line 5-6

    # Gradients of Gradients is kind of tricky!
    with tf.variable_scope('', reuse=True) as scope:
        d_hat = discriminator(x_hat)
        grad_D_x_hat = tf.gradients(d_hat, x_hat)[0]

    ## calculate penalty term in line 7
    grad_norm = tf.norm(grad_D_x_hat, axis=1)
    grad_pen = lam * tf.reduce_mean(tf.square(grad_norm - 1.0))

    D_loss = D_loss + grad_pen  ## calculate L^(i) following line 7

    return D_loss, G_loss





def get_solvers(learning_rate=1e-3, beta1=0.5):
    D_solver = None
    G_solver = None

    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

    return D_solver, G_solver


## Putting it all together now
tf.reset_default_graph()

batch_size = 128
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')

D_solver,G_solver = get_solvers()
D_loss, G_loss = wgangp_loss(logits_real, logits_fake, 128, x, G_sample)
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,
              show_every=250, print_every=50, batch_size=64, num_epoch=10):
    # compute the number of iterations we need
    max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
    for it in range(max_iter):
        # every show often, show a sample result

        '''
        if it % show_every == 0:
            samples = sess.run(G_sample)
            fig = show_images(samples[:16])
            plt.show()
            print()
            '''
        # run a batch of data through the network
        minibatch, minbatch_y = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it, D_loss_curr, G_loss_curr))
    #print('Final images')
    samples = sess.run(G_sample)
    return samples

    #fig = show_images(samples[:16])
    #plt.show()


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,
              batch_size=128, num_epoch=5)

print(samples.shape)
np.genfromtxt("samples.txt",samples)

