import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def binary_cross_entropy(x, y):
    zero = 1e-15
    return (-(x * tf.log(y + zero) + (1. - x) * tf.log(1. -y + zero)))


def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="inputs_z")
    
    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.01):
    ''' Build the generator network.
    
        Arguments
        ---------
        z : Input tensor for the generator
        out_dim : Shape of the generator output
        n_units : Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky reset_default_graph
        
        Returns
        -------
        out, logits: 
    '''

    #changed
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # Hidden layer
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1)
        
        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.nn.tanh(logits)
        
        return out#, logits


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    ''' Build the discriminator network.
    
        Arguments
        ---------
        x : Input tensor for the discriminator
        n_units: Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU
        
        Returns
        -------
        out, logits: 
    '''

    #changed
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1)
        
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.nn.sigmoid(logits)
        
        return out#, logits

# Size of input image to discriminator
input_size = 784 # 28x28 MNIST images flattened
# Size of latent vector to generator
z_size = 100
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Label smoothing 
smooth = 0
w=2

tf.reset_default_graph()
# Create our input placeholders

input_real, input_z = model_inputs(input_size, z_size)

g_model_array = []
d_model_real_array = []
d_model_fake_array = []
g_vars_array = []
d_vars_array = []
d_labels_real_array = []
d_labels_fake_array = []
d_loss_real_array = []
d_loss_fake_array = []
d_loss_array = []
g_loss_array = []

for i in range(w):
    # Generator network here
    g_model_array.append(generator(input_z, input_size, g_hidden_size, reuse=False,  alpha=alpha)) 
    # g_model is the generator output

    # Disriminator network here
    d_model_real_array.append(discriminator(input_real, d_hidden_size, reuse=False, alpha=alpha))
    d_model_fake_array.append(discriminator(g_model_array[i], d_hidden_size, reuse=True, alpha=alpha))

    # Calculate losses
    d_labels_real_array.append(tf.ones_like(d_model_real_array[i]) * (1 - smooth))
    d_labels_fake_array.append(tf.zeros_like(d_model_fake_array[i]))

    d_loss_real_array.append(binary_cross_entropy(d_labels_real_array[i], d_model_real_array[i]))
    d_loss_fake_array.append(binary_cross_entropy(d_labels_fake_array[i], d_model_fake_array[i]))

    d_loss_array.append(tf.reduce_mean(d_loss_real_array[i] + d_loss_fake_array[i]))

    g_loss_array.append(tf.reduce_mean(
        binary_cross_entropy(
            tf.ones_like(d_model_fake_array[i]), 
            d_model_fake_array[i])))

    # Get the trainable_variables, split into G and D parts
    g_vars_array.append([var for var in tf.trainable_variables() if var.name.startswith("generator")])
    d_vars_array.append([var for var in tf.trainable_variables() if var.name.startswith("discriminator")])

d_loss = d_loss_array[0]
for i in range(1,w):
    d_loss=d_loss_array[i]+d_loss
d_loss=d_loss/w

g_loss = g_loss_array[0]
for i in range(1,w):
    g_loss=g_loss_array[i]+g_loss
g_loss=g_loss/w

# Optimizers
learning_rate = 0.002
d_train_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars_array[w-1])
g_train_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars_array[w-1])


batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars_array[w-1])
with tf.Session() as sess:
    #filewriter = tf.summary.FileWriter('graph',sess.graph)
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            l=e
            r=e-w+1
            if(r<0):
                r=0

            d_new_fake_loss_array=[0]*batch_size
            d_new_real_loss_array=[0]*batch_size
            d_final_loss=0
            g_final_loss=0

            for i in range(l,r,-1):

                # Run optimizers
                d_new_loss,d_new_fake_loss,d_new_real_loss = sess.run([d_loss_array[i%w],d_loss_fake_array[i%w],d_loss_real_array[i%w]],feed_dict={input_real: batch_images, input_z: batch_z})

                g_new_loss = sess.run(g_loss_array[i%w],feed_dict={input_z: batch_z})

                d_new_real_loss_array = [d_new_real_loss_array[j]+d_new_real_loss[j] for j in range(batch_size)]
                d_new_fake_loss_array = [d_new_fake_loss_array[j]+d_new_fake_loss[j] for j in range(batch_size)]
                g_final_loss=g_final_loss+g_new_loss
                d_final_loss=d_final_loss+d_new_loss

            d_new_fake_loss_array[:] = [i/w for i in d_new_fake_loss_array]
            d_new_real_loss_array[:] = [i/w for i in d_new_real_loss_array]
            
            d_final_loss=d_final_loss/w
            g_final_loss=g_final_loss/w

            d_new_real_loss_array = np.mean(d_new_real_loss_array)
            d_new_fake_loss_array = np.mean(d_new_fake_loss_array)

            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # At the end of each epoch, get the losses and print them out
        #train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        #train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(d_final_loss),
              "Generator Loss: {:.4f}".format(g_final_loss))    
        # Save losses to view after training
        losses.append((d_final_loss, g_final_loss))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, reuse=True),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
