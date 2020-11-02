import os
import numpy as np
import tensorflow as tf
import source.layers as lay

class GANomaly(object):

    def __init__(self, \
        height, width, channel, ksize, \
        w_enc=1, w_con=50, w_adv=1, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel, self.ksize = height, width, channel, ksize
        self.w_enc, self.w_con, self.w_adv = w_enc, w_con, w_adv
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[])
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[])

        self.layer = lay.Layers()

        self.conv_shapes = []
        self.variables, self.losses = {}, {}
        self.__build_model(x_real=self.x, ksize=self.ksize, verbose=verbose)
        self.__build_loss()

        with tf.control_dependencies(tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate).minimize(self.losses['target'])

        tf.compat.v1.summary.scalar('GANomaly/loss_enc', self.losses['mean_enc'])
        tf.compat.v1.summary.scalar('GANomaly/loss_con', self.losses['mean_con'])
        tf.compat.v1.summary.scalar('GANomaly/loss_adv', self.losses['mean_adv'])
        tf.compat.v1.summary.scalar('GANomaly/loss_target', self.losses['target'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, iteration=0, training=False):

        feed_tr = {self.x:x, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.batch_size:x.shape[0], self.training:False}

        summaries = None
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
            except:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr)
            self.summary_writer.add_summary(summaries, iteration)

        x_fake, loss_enc, loss_con, loss_adv, loss_tot = \
            self.sess.run([self.variables['x_fake'], self.losses['mean_enc'], self.losses['mean_con'], self.losses['mean_adv'], self.losses['target']], \
            feed_dict=feed_te)

        outputs = {'x_fake':x_fake, \
            'loss_enc':loss_enc, 'loss_con':loss_con, 'loss_adv':loss_adv, 'loss_tot':loss_tot, \
            'summaries':summaries}
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def confirm_bn(self, verbose=True):

        print("\n* Confirm Batch Normalization")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            if('bn' in var.name):
                tmp_x = np.zeros((1, self.height, self.width, self.channel))
                values = self.sess.run(var, \
                    feed_dict={self.x:tmp_x, self.batch_size:1, self.training:False})
                if(verbose): print(var.name, var.shape)
                if(verbose): print(values)

    def loss_l1(self, a, b, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.abs(a - b), axis=reduce)

        return distance

    def loss_l2(self, a, b, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.sqrt(\
            tf.math.square(a - b) + 1e-9), axis=reduce)

        return distance

    def __init_session(self, path):

        try:
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

    def __build_loss(self):

        # Loss 1: Encoding loss (L2 distance)
        loss_enc = self.loss_l2(self.variables['z_real'], self.variables['z_fake'], [1, 2, 3])
        # Loss 2: Restoration loss (L1 distance)
        loss_con = self.loss_l1(self.x, self.variables['x_fake'], [1, 2, 3])
        # Loss 3: Adversarial loss (L2 distance)
        loss_adv = self.loss_l2(self.variables['d_real'], self.variables['d_fake'], [1, 2, 3])

        for fidx, _ in enumerate(self.variables['f_real']):
            feat_dim = len(self.variables['f_real'][fidx].shape)
            if(feat_dim == 4):
                loss_adv += self.loss_l2(self.variables['d_real'], self.variables['d_fake'], [1, 2, 3])
            elif(feat_dim == 3):
                loss_adv += self.loss_l2(self.variables['d_real'], self.variables['d_fake'], [1, 2])
            elif(feat_dim == 2):
                loss_adv += self.loss_l2(self.variables['d_real'], self.variables['d_fake'], [1])
            else:
                loss_adv += self.loss_l2(self.variables['d_real'], self.variables['d_fake'])

        self.losses['mean_enc'] = tf.compat.v1.reduce_mean(loss_enc)
        self.losses['mean_con'] = tf.compat.v1.reduce_mean(loss_con)
        self.losses['mean_adv'] = tf.compat.v1.reduce_mean(loss_adv)

        self.losses['loss_enc'] = loss_enc
        self.losses['loss_con'] = loss_con
        self.losses['loss_adv'] = loss_adv

        self.losses['target'] = tf.compat.v1.reduce_mean(\
            self.losses['loss_enc'] * self.w_enc\
            + self.losses['loss_con'] * self.w_con\
            + self.losses['loss_adv'] * self.w_adv)

    def __build_model(self, x_real, ksize=3, verbose=True):

        if(verbose): print("\n* Encoder")
        self.variables['z_real'], _ = \
            self.__encoder(x=x_real, ksize=ksize, reuse=False, \
            name='enc', verbose=verbose)
        if(verbose): print("\n* Decoder")
        self.variables['x_fake'] = \
            self.__decoder(z=self.variables['z_real'], ksize=ksize, reuse=False, \
            name='dec', verbose=verbose)
        self.variables['z_fake'], _ = \
            self.__encoder(x=self.variables['x_fake'], ksize=ksize, reuse=True, \
            name='enc', verbose=False)

        if(verbose): print("\n* Discriminator")
        self.variables['d_real'], self.variables['f_real'] = \
            self.__encoder(x=x_real, ksize=ksize, reuse=False, \
            name='dis', verbose=verbose)
        self.variables['d_fake'], self.variables['f_fake'] = \
            self.__encoder(x=self.variables['x_fake'], ksize=ksize, reuse=True, \
            name='dis', verbose=False)

    def __encoder(self, x, ksize=3, reuse=False, name='enc', activation='lrelu', verbose=True):

        with tf.variable_scope(name, reuse=reuse):
            featurebank = []

            conv1_1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 1, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_1" %(name), verbose=verbose)
            if('dis' in name): featurebank.append(conv1_1)
            conv1_2 = self.layer.conv2d(x=conv1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_2" %(name), verbose=verbose)
            if('enc' in name and not(reuse)): self.conv_shapes.append(conv1_2.shape)
            if('dis' in name): featurebank.append(conv1_2)
            maxp1 = self.layer.maxpool(x=conv1_2, ksize=2, strides=2, padding='SAME', \
                name="%s_pool1" %(name), verbose=verbose)

            conv2_1 = self.layer.conv2d(x=maxp1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_1" %(name), verbose=verbose)
            if('dis' in name): featurebank.append(conv2_1)
            conv2_2 = self.layer.conv2d(x=conv2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_2" %(name), verbose=verbose)
            if('enc' in name and not(reuse)): self.conv_shapes.append(conv2_2.shape)
            if('dis' in name): featurebank.append(conv2_2)
            maxp2 = self.layer.maxpool(x=conv2_2, ksize=2, strides=2, padding='SAME', \
                name="%s_pool2" %(name), verbose=verbose)

            conv3_1 = self.layer.conv2d(x=maxp2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_1" %(name), verbose=verbose)
            if('dis' in name): featurebank.append(conv3_1)
            conv3_2 = self.layer.conv2d(x=conv3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_2" %(name), verbose=verbose)
            if('enc' in name):
                e = conv3_2
            else:
                e = self.layer.activation(x=conv3_2, activation='sigmoid', name="%s_fin" %(name))
            if('enc' in name and not(reuse)): self.conv_shapes.append(e.shape)
            if('dis' in name): featurebank.append(e)

            return e, featurebank

    def __decoder(self, z, ksize=3, reuse=False, name='dec', activation='lrelu', verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            convt1_1 = self.layer.conv2d(x=z, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_1" %(name), verbose=verbose)
            convt1_2 = self.layer.conv2d(x=convt1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_2" %(name), verbose=verbose)

            [n, h, w, c] = self.conv_shapes[-2]
            convt2_1 = self.layer.convt2d(x=convt1_2, stride=2, padding='SAME', \
                output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 32, 64], \
                dilations=[1, 1, 1, 1], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_1" %(name), verbose=verbose)
            convt2_2 = self.layer.conv2d(x=convt2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_2" %(name), verbose=verbose)

            [n, h, w, c] = self.conv_shapes[-3]
            convt3_1 = self.layer.convt2d(x=convt2_2, stride=2, padding='SAME', \
                output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 16, 32], \
                dilations=[1, 1, 1, 1], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_1" %(name), verbose=verbose)
            convt3_2 = self.layer.conv2d(x=convt3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_2" %(name), verbose=verbose)
            d = self.layer.conv2d(x=convt3_2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, self.channel], batch_norm=True, training=self.training, \
                activation="sigmoid", name="%s_conv3_3" %(name), verbose=verbose)

            return d
