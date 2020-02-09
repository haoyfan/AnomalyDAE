from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)
        # self.z_mean = self.embeddings

        # decoder1
        self.attribute_decoder_layer1 = GraphConvolution(input_dim=FLAGS.hidden2,
                                           output_dim=FLAGS.hidden1,
                                           adj=self.adj,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.embeddings)

        self.attribute_decoder_layer2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=self.input_dim,
                                               adj=self.adj,
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               logging=self.logging)(self.attribute_decoder_layer1)

        # decoder2
        self.structure_decoder_layer1 = GraphConvolution(input_dim=FLAGS.hidden2,
                                           output_dim=FLAGS.hidden1,
                                           adj=self.adj,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.embeddings)

        self.structure_decoder_layer2 = InnerProductDecoder(input_dim=FLAGS.hidden1,
                                        act=tf.nn.sigmoid,
                                        logging=self.logging)(self.structure_decoder_layer1)


        self.attribute_reconstructions = self.attribute_decoder_layer2
        self.structure_reconstructions = self.structure_decoder_layer2


class AnomalyDAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,
                 decoder_act=[tf.nn.sigmoid, tf.nn.sigmoid], **kwargs):
        super(AnomalyDAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.decoder_act = decoder_act
        self.build()

    def _build(self):
        self.hidden1 = Dense(input_dim=self.input_dim,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.relu,
                             sparse_inputs=True,
                             dropout=self.dropout)(self.inputs)

        self.hidden1 = tf.expand_dims(self.hidden1, 1)
        attns = []
        k=1
        for _ in range(k):
            attns.append(NodeAttention(bias_mat=self.adj, nb_nodes=self.n_samples,
                                       # act=tf.nn.relu,
                                       act=lambda x: x,
                                       out_sz=FLAGS.hidden2//k)(self.hidden1))

        self.embeddings_s = tf.concat(attns, axis=-1)[0]

        self.hidden2 = Dense(input_dim=self.n_samples,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.relu,
                             sparse_inputs=True,
                             dropout=self.dropout)(tf.sparse_transpose(self.inputs))

        self.embeddings_a = Dense(input_dim=FLAGS.hidden1,
                              output_dim=FLAGS.hidden2,
                              act=lambda x: x,
                              # act=tf.nn.relu,
                              dropout=self.dropout)(self.hidden2)
        print("FLAGS.hidden2,",FLAGS.hidden2)

        self.structure_reconstructions, self.attribute_reconstructions\
            = InnerDecoder(input_dim=FLAGS.hidden2,
                                            act=self.decoder_act,
                                            logging=self.logging)((self.embeddings_s, self.embeddings_a))


