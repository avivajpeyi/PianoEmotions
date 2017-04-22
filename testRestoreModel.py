
import tensorflow as tf
import numpy as np
import pickle


# Recreate the EXACT SAME variables
#################

saveFile = "savedModels/musicModel"


n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10



x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([128, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

#################


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('savedModels/musicModel.meta')
  new_saver.restore(sess, 'savedModels/musicModel')


#### CODE NEEDED
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     new_saver = tf.train.import_meta_graph('savedModels/musicModel.meta')
#     new_saver.restore(sess, 'savedModels/musicModel')

print("restored my model")
