# v4 : 1. Follow Tensorflow demo code methodology,
#         * Embedding crop + random noise + standardization into training
#         * Embedding crop or pad + standardization into training
#         * Using tf.cond to branch two input paths instead of using two sessions.
#      2. Change tf.truncated_normal to tf.contrib.layers.xavier_initializer()

# v5 : 1. Try manually controlled decay learning rate

# +---+----+---+--+----+---+-+----+---+-+-+-----+----++---+----+---+--+----+---+-+----+---+-+-+---+
# Graph + Name Scope + Loss Function + Optimizer + Batch
# +---+----+---+--+----+---+-+----+---+-+-+-----+----++---+----+---+--+----+---+-+----+---+-+-+---+

# ----------------------------------------------------------------------------------------
# 1. Created the data.
# 2. Initialized placeholders and variables.
# 3. Created a loss function.
# 4. Choose an optimizer.
# 5. And finally, iterated across random data samples to iteratively update our variables.
# ----------------------------------------------------------------------------------------



import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

from tf_common_layers_v3 import *

# Try Lock Random Seed

seed = 7+40+14+42+45
tf.set_random_seed(seed)
np.random.seed(seed)



# CIFAR-10

# Understand Cifar-10 and Cifar-100 dataset
# (R) www.cs.toronto.edu/~kriz/cifar.html

import os

# (!) Use Example code from https://docs.python.org/3/library/tarfile.html
def untar(file, tgtPath = '.\\cifar-10'):
    import tarfile
    tar = tarfile.open(file)
    tar.extractall(path=tgtPath)
    tar.close()


# (!) Use the extraction code provided
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# <--- Cifar-10 --->
cifar10_path = 'cifar-10-python.tar.gz'
cifar10_extraction_path = '.\\cifar-10'
cifar10_already_extracted = os.path.exists(cifar10_extraction_path)
if(not cifar10_already_extracted):
    untar( cifar10_path, cifar10_extraction_path )

# < DE-SERIALIZE DATA >
cifar10_meta = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\batches.meta' ) #
cifar10_batch1 = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\data_batch_1' ) #
cifar10_batch2 = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\data_batch_2' ) #
cifar10_batch3 = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\data_batch_3' ) #
cifar10_batch4 = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\data_batch_4' ) #
cifar10_batch5 = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\data_batch_5' ) #
cifar10_test = unpickle( cifar10_extraction_path + '\\cifar-10-batches-py' + '\\test_batch' ) #


data = tf.concat( [cifar10_batch1[b'data'], cifar10_batch2[b'data'], 
                   cifar10_batch3[b'data'], cifar10_batch4[b'data'], 
                   cifar10_batch5[b'data']] , axis = 0 )
label = tf.concat( [cifar10_batch1[b'labels'], cifar10_batch2[b'labels'], 
                    cifar10_batch3[b'labels'], cifar10_batch4[b'labels'], 
                    cifar10_batch5[b'labels']] , axis = 0 )

test_data, test_label = cifar10_test[b'data'], cifar10_test[b'labels']

crop_width, crop_height = 24, 24

def training_preprocess(image):
    image = tf.random_crop(image, [crop_height, crop_width, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(image)

def train_standardization(images):
    return tf.map_fn(training_preprocess, images)
    # return tf.stack([training_preprocess(x) for x in tf.unstack(images)])  # (X) tf.unstack() cannot work with None

def testing_preprocess(image):
    resized_image = tf.image.resize_image_with_crop_or_pad( image, crop_height, crop_width)
    return tf.image.per_image_standardization(resized_image)

def eval_standardization(images):
    return tf.map_fn(testing_preprocess, images)
    # return tf.stack([testing_preprocess(x) for x in tf.unstack(images)])  # (X) tf.unstack() cannot work with None


# Model
class model1:
    def __init__( self, image_height, image_width, inChannels, outLabels ):
        # g = tf.Graph()
        # self.g = g
        # self.g = tf.Graph() (!) TRY THIS LATER !!!!!!!!!!!!!!!!!!!
        self.image_height, self.image_width, self.inChannels = image_height, image_width, inChannels
        self.g = tf.get_default_graph()
        with self.g.as_default():
			# !!! ALL GRAPH-RELATED STAYS INSIDE THIS SCOPE !!!
			#     ( including initializer-op )
            self.buildGraph(image_height, image_width, inChannels, outLabels)
            init = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.g)
        # WRONG POSITION : init = tf.global_variables_initializer()
        self.sess.run(init)
    def __del__( self ):
        self.sess.close()
    def sess_init(self):
        with self.g.as_default():
            init = tf.global_variables_initializer()
            # self.sess = tf.Session(graph = self.g)
            self.sess.run(init)
        self.sess.run(self.iter_init_op)
        self.sess.run(self.test_iter_init_op)
        print('sess init()')
    def buildGraph( self, image_height, image_width, inChannels, outLabels ):
        # with self.g.as_default():
        self.net = {}
        with tf.name_scope('InputLayers') as InputLayer:
            self.net['is_training'] = tf.placeholder(tf.bool)
            self.net['keep_prob'] = tf.placeholder(tf.float32)
            self.net['input'] = tf.placeholder(tf.float32, shape = [None, image_height * image_width * inChannels], name = 'input')
            # self.net['input'] = tf.placeholder(tf.float32, shape = [None, image_height * image_width * inChannels], name = 'input')
            self.net['label'] = tf.placeholder(tf.int32, shape = [None], name = 'label')
            self.net['input2'] = tf.reshape(self.net['input'], [-1, image_height, image_width, inChannels], name = 'input2')
            self.net['input3'] = tf.cond( self.net['is_training'], lambda : train_standardization(self.net['input2'])
                                                                 , lambda : eval_standardization(self.net['input2']) )
            # self.net['input2'] = tf.reshape(self.net['input'], [-1, image_height, image_width, inChannels], name = 'input2')
        
        with tf.name_scope('oooLayers') as ConvLayer:
            self.net['C1'] = conv2( BN(self.net['input3']), 32, 'Conv1')

        
        with tf.name_scope('xxxLayers') as FcLayers:
            '''
            self.net['FC1'] = dense( self.net['M2'], 100 )
            self.net['Out'] = dense( self.net['FC1'], outLabels, activation = tf.identity )
            '''
            def to1d(x) :
                length = np.prod(x.get_shape().as_list()[1:]) 
                return tf.reshape(x, [-1, length])

            self.net['Out'] = tf.layers.dense( self.net['FCX'], outLabels )
            #self.net['SFMX'] = softmax( self.net['FC4'] )
        with tf.name_scope('Loss') as lossFn:
            # self.net['loss'] = tf.reduce_mean( tf.cast( tf.argmax( self.net['SFMX'], axis = -1), tf.int32 ) - self.net['label'] )
            # self.net['loss'] = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.net['label'], logits=self.net['Out']) )
            self.loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.net['label'], logits=self.net['Out']) )
            # tf.summary.scalar(self.net['loss'])
        print('build graph : ', self.g, tf.get_default_graph())
        # self.dropoutRate = tf.placehoder(tf.float32, shape = [])
        self.net['learningRate'] = tf.placeholder(tf.float32, shape = [])
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.net['learningRate']) 
        # self.net['train_op'] = self.optimizer.minimize( self.net['loss'] )
        self.net['train_op'] = self.optimizer.minimize( self.loss )
        tf.summary.scalar('loss', self.loss)

        self.setLearningRate()
        self.setKeepProb()
        self.merged = tf.summary.merge_all()

    def setLearningRate(self, newLearningRate = 0.01):
        self.learniungRate = newLearningRate
    def setKeepProb(self, newKeepProb = 1.0):
        self.keepProb = newKeepProb

    def graph(self):
        return self.g
        # Or : return self.sess.graph

    def debug(self, layerName):
        return self.sess.run(net[layerName])

    def __call__( self, inData ):
        # Inference
        feed_dict = { self.net['input'] : inData, self.net['keep_prob'] : 1.0  }
        return self.sess.run( tf.argmax(self.net['Out'], 1), feed_dict=feed_dict )

    def train( self ):
        # training
        data, label = self.sess.run( [ self.data_el, self.label_el ] )
        # (X) Cannot feed data pass-thru tf. operations, must do tf operation inside !!!
        # data = tf.reshape(data, [-1, self.image_height, self.image_width, self.inChannels])
        # data = tf.cast(data, tf.float32)
        # data = train_standardization(data)
        feed_dict = { self.net['input'] : data, 
                      self.net['label'] : label, 
                      self.net['learningRate'] : self.learniungRate,
                      self.net['keep_prob'] : self.keepProb,
                      self.net['is_training'] : np.array(True) }
        return self.sess.run( [self.loss, self.net['train_op']], feed_dict=feed_dict )

    def test( self ):
        x, y = self.sess.run( [self.tdata_el, self.tlabel_el] )
        # x = tf.reshape(x, [-1, self.image_height, self.image_width, self.inChannels])
        # x = tf.cast(x, tf.float32)
        # x = eval_standardization(x)
        feed_dict = { self.net['input'] : x, self.net['keep_prob'] : 1.0, self.net['is_training'] : np.array(False) } # (X) Cannot use tf.constant(False)
        return self.sess.run( tf.argmax( tf.nn.softmax(self.net['Out']), 1), feed_dict = feed_dict ), y

    def import_dataset( self, inData, BATCH_SIZE = 128 ):
        data, label = inData
        # with self.g.as_default():
        with self.g.as_default():
            print(1, self.g, tf.get_default_graph())
            with tf.variable_scope('dataset'):
                print(1.5, self.g, tf.get_default_graph())
                self.dataset = tf.data.Dataset.from_tensor_slices( (data, label) )
                self.dataset = self.dataset.repeat() # With repeat(None) as default means repeat INDEFINITELY
                self.dataset = self.dataset.shuffle(buffer_size=500)
                self.dataset = self.dataset.batch(BATCH_SIZE)
                self.iter = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                            self.dataset.output_shapes)
                self.iter_init_op = self.iter.make_initializer(self.dataset)
                self.data_el, self.label_el = self.iter.get_next()
    def import_testset( self, inData, BATCH_SIZE = 128 ):
        test_data, test_label = inData
        # with self.g.as_default():
        testset = tf.data.Dataset.from_tensor_slices( (test_data, test_label) )
        testset = testset.repeat()
        testset = testset.shuffle(buffer_size=500)
        testset = testset.batch(BATCH_SIZE)
        self.testset = testset
        # self.test_iter = self.testset.make_one_shot_iterator()
        
        with self.g.as_default():
            self.test_iter = tf.data.Iterator.from_structure(self.testset.output_types,
                                                        self.testset.output_shapes)
            self.tdata_el, self.tlabel_el = self.test_iter.get_next()
            with tf.variable_scope('testset'):
                self.test_iter_init_op = self.test_iter.make_initializer(self.testset)



# The Ops

BATCH_SIZE = 500

model = model1( 32, 32, 3, 10 )

writer = tf.summary.FileWriter('.\\logs', model.graph())

model.import_dataset( (data, label), BATCH_SIZE = BATCH_SIZE )
model.import_testset( (test_data, test_label), BATCH_SIZE = BATCH_SIZE )

model.sess_init()

def init():
    model.sess_init()

def train( lr = 0.001, kb = 0.8, EPOCH = 1000, SHOW_LOSS_EVERY = 10 ):
    model.setLearningRate(lr)
    model.setKeepProb(kb)
    loss = None
    for i in range( EPOCH ):
        loss, _ = model.train()
        if i % SHOW_LOSS_EVERY == 0:
        	print( i, loss )
            # (X) NOT WRITING SUMMARY YET !!!

def test():
	pred, label = model.test()
	accuracy = float(np.sum(np.equal(pred, label))) / len(pred)
	print(pred, label, accuracy)

# USAGE :
# import toClass_v3 as t0
test()
# t0.init()
# train(EPOCH = 1000, lr=0.0001)


train(EPOCH=200, lr =0.1, kb = 0.5)

