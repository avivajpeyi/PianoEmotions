import tensorflow as tf
import pickle
import numpy as np


n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
hm_data = 2000000

batch_size = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')


current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([278, n_nodes_hl1])),
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



def neural_network_model(data):
    ####INPUT LAYER (HIDDEN LAYER 1)
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    ####HIDDEN LAYER 2
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    ####HIDDEN LAYER 3
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    ####OUTPUT LAYER
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

saver = tf.train.Saver()

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('musicModel.pickle','rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"model.ckpt")

        #### CONVERT THE MIDI TO NOTES AND FEATURES (without [0,1])
        #### need it in the [0 112 1 1 0 0 0 ....] format
        


        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Happy Notes:',input_data)
        elif result[0] == 1:
            print('Sad Notes:',input_data)


# opne midi file
#use_neural_network(pass midi file)
