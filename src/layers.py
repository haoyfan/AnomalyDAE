from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class FullyConnectedDecoder(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(FullyConnectedDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        outputs = tf.matmul(x, self.vars['weights'])
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        #x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.sparse_inputs:
            output = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        else:
            output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)




class NodeAttention(Layer):
    """Dense layer."""

    def __init__(self, out_sz, bias_mat, nb_nodes, dropout=0.,
                 act=tf.nn.elu, **kwargs):
        super(NodeAttention, self).__init__(**kwargs)
        self.act = act
        self.out_sz = out_sz
        self.bias_mat = bias_mat
        self.nb_nodes = nb_nodes

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        seq_fts = tf.layers.conv1d(inputs, self.out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1_t = tf.layers.conv1d(seq_fts, 1, 1)
        f_2_t = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1_t, (self.nb_nodes, 1))
        f_2 = tf.reshape(f_2_t, (self.nb_nodes, 1))

        f_1 = self.bias_mat * f_1
        f_2 = self.bias_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [self.nb_nodes, self.nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.out_sz])
        ret = self.act(tf.contrib.layers.bias_add(vals))

        return ret  # activation
        # return ret, inputs, seq_fts, f_1_t, f_2_t, logits


class InnerDecoder(Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=[tf.nn.sigmoid, tf.nn.sigmoid], **kwargs):
        super(InnerDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.input_dim = input_dim
        self.act_struc = act[0]
        self.act_attr = act[1]


    def _call(self, inputs):
        z_u, z_a = inputs
        z_u = tf.nn.dropout(z_u, 1 - self.dropout)
        z_u_t = tf.transpose(z_u)
        x = tf.matmul(z_u, z_u_t)

        z_a_t = tf.transpose(tf.nn.dropout(z_a, 1 - self.dropout))
        y = tf.matmul(z_u, z_a_t)

        edge_outputs = self.act_struc(x)
        attri_outputs = self.act_attr(y)
        return edge_outputs, attri_outputs
