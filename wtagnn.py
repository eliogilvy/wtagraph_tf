import math
import tensorflow as tf
from tensorflow import keras
from keras import layers

g = None

def wtagnn_msg(edge):
    nb_ef = edge.data['ef']
    return {'nb_ef': nb_ef}

def wtagnn_reduce(node):
    nb_ef = tf.reduce_mean(node.mailbox['nb_ef'], 1)
    return {'nb_ef':nb_ef}

class NodeApplyModule(layers.Layer):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()

        self.out_feats = out_feats
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,), dtype='float32'), trainable=True)
        else:
            bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.bias.shape[0])
        b_init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        self.bias = tf.Variable(initial_value=b_init(shape=(self.out_feats,), dtype = 'float32'), trainable=True)

    def call(self, nodes):
        nf = nodes.data['nf']
        nb_ef = nodes.data['nb_ef']

        if self.bias is not None:
            nf = nf + self.bias
        if self.activation:
            nf = self.activation(nf)
        return {'nf': nf, 'nb_ef': nb_ef}

class EdgeAppleModule(layers.Layer):
    def __init__(self, out_feats, activation=None, bias=True):
        super(EdgeAppleModule, self).__init__()
        self.out_feats = out_feats
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,), dtype='float32'), trainable=True)
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()
        self.my_dense = layers.Dense(out_feats)
    
    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.shape[0])
            b_init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            self.bias = tf.Variable(initial_value=b_init(shape=(self.out_feats,), dtype='float32'), trainable=True)
    
    def call(self, edges):
        tmp_ef = edges.data['ef'] + edges.dst['nb_ef']
        srcdst_nf = (edges.src['nf'] + edges.dst['nf']) / 2
        ef = self.my_dense(tf.concat([tmp_ef, srcdst_nf], 1))

        if self.bias is not None:
            ef = ef + self.bias
        if self.activation:
            ef = self.activation(ef)
        return {'ef': ef}      

class WTAGNNLayer(layers.Layer):
    def __init__(self, g, in_feats_node, in_feats_edge, out_feats, activation,
                 dropout, bias=True):
        super(WTAGNNLayer, self).__init__()
        self.g = g
        self.in_feats_node = in_feats_node
        self.in_feats_edge = in_feats_edge
        self.out_feats = out_feats

        w_init = tf.random_normal_initializer()
        self.weight_node = tf.Variable(initial_value=w_init(shape=(in_feats_node, out_feats), dtype = 'float32'), trainable=True)
        self.weight_edge = tf.Variable(initial_value=w_init(shape=(in_feats_edge, out_feats), dtype = 'float32'), trainable=True)

        if dropout:
            self.dropout = layers.Dropout(rate=dropout)
        else:
            self.dropout = 0
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.edge_update = EdgeAppleModule(out_feats, activation, bias)
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_node.shape[1])
        w_init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        self.weight_node = tf.Variable(initial_value=w_init(shape=(self.in_feats_node, self.out_feats),dtype='float32'), trainable=True)

        stdv = 1. / math.sqrt(self.weight_edge.shape[1])
        w_init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        self.weight_edge = tf.Variable(initial_value=w_init(shape=(self.in_feats_edge, self.out_feats), minval=-stdv, maxval=stdv, dtype='float32'), trainable=True)
    
    def call(self, nf, ef):
        if self.dropout:
            nf = self.dropout(nf)
        self.g.ndata['nf'] = tf.matmul(nf, self.weight_node)
        self.g.edata['ef'] = tf.matmul(ef, self.weight_edge)

        global g
        g = self.g
        self.g.update_all(wtagnn_msg, wtagnn_reduce)

        self.g.apply_nodes(func=self.node_update)
        self.g.apply_edges(func=self.edge_update)

        nf = self.g.ndata.pop('nf')
        ef = self.g.edata.pop('ef')
        self.g.ndata.pop('nb_ef')

        return nf, ef

class WTAGNN(layers.Layer):
    def __init__(self, g,  input_node_feat_size, input_edge_feat_size, n_hidden, n_classes,
                 n_layers,  activation, dropout):
        super(WTAGNN, self).__init__()
        self.layers = []
        self.layers.append(WTAGNNLayer(g, input_node_feat_size, input_edge_feat_size, n_hidden, activation, dropout))
        for i in range(n_layers - 1):
            self.layers.append(WTAGNNLayer(g, n_hidden, n_hidden, n_hidden, activation, dropout))
        self.layers.append(WTAGNNLayer(g, n_hidden, n_hidden, n_classes, None, dropout))
    
    def call(self, nf, ef):
        for layer in self.layers:
            nf, ef = layer(nf, ef)
        return nf, ef