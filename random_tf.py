# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:20:57 2018

@author: tommy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

true_theta = 2.
true_theta_0 = .5

x=np.linspace(0,1,1000)
y=true_theta*x + true_theta_0 + np.random.randn(x.size)*0.3

x = x.reshape((-1,1))
y = y.reshape((-1,1))

class Model(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32,shape=[None,1],name='input')
        self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        
    def denselayer_with_dropout(self, input_, out_dim , keep_prob):
        '''
        dense = tf.layers.dense(inputs=input_, units= out_dim , use_bias = False,
                                activation=tf.nn.relu)#tf.nn.relu
        return dense
        '''
        w = tf.Variable(tf.random_normal([]), name='weight')
        b = tf.Variable(tf.random_normal([]), name='bias')
        self.w=w
        self.b=b
        nonlin_model = tf.add(tf.multiply(input_, w), b)
        return nonlin_model
        
    def inference(self):
        dense_out = self.denselayer_with_dropout(self.input,1,1)
        return dense_out
    
    def build_optimizer(self,loss_val,trainable_var,learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate) 
        grads = optimizer.compute_gradients(loss_val, var_list=trainable_var)
        return optimizer.apply_gradients(grads,global_step=self.global_step)
        
    def build_train(self,learning_rate):
        raw_output = self.inference()
        rawloss = tf.losses.mean_squared_error(self.label,raw_output)
        loss = tf.reduce_mean(rawloss)
        trainable_var = tf.trainable_variables()
        train_op = self.build_optimizer(loss,trainable_var,learning_rate)
        return train_op,loss,raw_output
    
    def before_session_initialization(self,learning_rate):
        self.train_op,self.loss_op, self.raw_out = self.build_train(
            learning_rate=learning_rate)
        
    def after_session_initialization(self,sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        self.train_writer = tf.summary.FileWriter('./', sess.graph)
        
    def train(self,sess,data,label):
        for i in range(1000):
            batch_data = data
            batch_label = label
            feed_dict = {self.input: batch_data, self.label: batch_label}
            _, train_loss,out= sess.run([self.train_op,self.loss_op,self.raw_out], feed_dict=feed_dict)
        print(train_loss)
            #print(out)
            #print(batch_label)
            
    def test(self,sess,data):
        feed_dict = {self.input: data}
        out= sess.run([self.raw_out], feed_dict=feed_dict)
        return out
        
    def vb (self,sess):
        print(sess.run(self.w))
        print(sess.run(self.b))

tf.reset_default_graph()
model = Model()
model.before_session_initialization(learning_rate=0.5)
sess = tf.InteractiveSession()
model.after_session_initialization(sess)
model.train(sess,x,y)
out = model.test(sess,x)
model.vb(sess)
sess.close()
plt.scatter(x,y)
#print(out[0].shape)
plt.plot(x,out[0],color='r')
#plt.show()
