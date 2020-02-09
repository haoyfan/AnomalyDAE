import scipy.sparse as sp
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data2(data_source):
    data = scipy.io.loadmat("../data/{}/{}.mat".format(data_source,data_source))
    labels = data["Label"]

    attr_ = data["Attributes"]
    attributes = sp.csr_matrix(attr_)
    network = sp.lil_matrix(data["Network"])

    return network, attributes, labels


def load_data(data_source):
    data = scipy.io.loadmat("../data/{}.mat".format(data_source))
    labels = data["gnd"]
    labels = data["Label"]

    attributes = sp.csr_matrix(data["X"])
    network = sp.lil_matrix(data["A"])

    return network, attributes, labels

def format_data(data_source):

    adj, features, labels = load_data2(data_source)

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features, labels]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]]
        item_name = retrieve_name(item)
        feas[item_name] = item

    return feas


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and "item" not in var_name][0]
