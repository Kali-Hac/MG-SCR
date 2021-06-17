import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

"""
 Generate training data for each dataset.
"""

def gen_train_data(dataset, split, time_step, nb_nodes, nhood, global_att, batch_size, view=''):
	def get_data(dimension, fr):
		input_data = np.load(
			'Datasets/' + frames_ps + dataset + '_train_npy_data/source_' + dimension + '_' + dataset + '_' + str(
				fr) + '.npy')
		input_data = input_data.reshape([-1, time_step, nb_nodes])
		spine_pos = input_data[:, :, 0]
		spine_pos = np.expand_dims(spine_pos, -1)
		input_data = input_data - spine_pos
		if dataset == 'IAS':
			t_input_data = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/t_source_' + dimension + '_IAS-' + split + '_' + str(
					fr) + '.npy')
		else:
			t_input_data = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/t_source_' + dimension + '_' + dataset + '_' + str(
					fr) + '.npy')
		t_input_data = t_input_data.reshape([-1, time_step, nb_nodes])
		# Normalize
		t_spine_pos = t_input_data[:, :, 0]
		t_spine_pos = np.expand_dims(t_spine_pos, -1)
		t_input_data = t_input_data - t_spine_pos

		return input_data, t_input_data

	if view == '':
		frames_ps = dataset + '/' + str(time_step) + '/'
	else:
		frames_ps = dataset + '/' + str(time_step) + '/view_' + str(view) + '/'
	input_data_x, t_input_data_x = get_data('x', fr=time_step)
	input_data_y, t_input_data_y = get_data('y', fr=time_step)
	input_data_z, t_input_data_z = get_data('z', fr=time_step)

	X_train = np.concatenate([input_data_x, input_data_y, input_data_z], axis=-1)
	X_test = np.concatenate([t_input_data_x, t_input_data_y, t_input_data_z], axis=-1)

	ids = np.load(
		'Datasets/' + frames_ps + dataset + '_train_npy_data/ids_' + dataset + '_' + str(time_step) + '.npy')
	ids = ids.item()
	if dataset == 'IAS':
		t_ids = np.load(
			'Datasets/' + frames_ps + 'IAS_test_npy_data/ids_IAS-' + split + '_' + str(time_step) + '.npy')
	else:
		t_ids = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/ids_' + dataset + '_' + str(time_step) + '.npy')
	t_ids = t_ids.item()

	y_train = np.load(
		'Datasets/' + frames_ps + dataset + '_train_npy_data/frame_id_' + dataset + '_' + str(time_step) + '.npy')
	if dataset == 'IAS':
		y_test = np.load(
			'Datasets/' + frames_ps + 'IAS_test_npy_data/frame_id_IAS-' + split + '_' + str(time_step) + '.npy')
	else:
		y_test = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/frame_id_' + dataset + '_' + str(time_step) + '.npy')

	X_train, y_train = class_samp_gen(X_train, y_train, ids, batch_size)
	# print(X_train.shape, y_train.shape)

	ids_keys = sorted(list(ids.keys()))
	classes = [i for i in ids_keys]
	y_train = label_binarize(y_train, classes=classes)
	t_ids_keys = sorted(list(t_ids.keys()))
	classes = [i for i in t_ids_keys]
	y_test = label_binarize(y_test, classes=classes)

	X_train_J = X_train.reshape([-1, time_step, 3, nb_nodes])
	X_train_J = np.transpose(X_train_J, [0, 1, 3, 2])
	X_train_P = reduce2part(X_train_J, nb_nodes)
	X_train_B = reduce2body(X_train_J, nb_nodes)

	X_test_J = X_test.reshape([-1, time_step, 3, nb_nodes])
	X_test_J = np.transpose(X_test_J, [0, 1, 3, 2])
	X_test_P = reduce2part(X_test_J, nb_nodes)
	X_test_B = reduce2body(X_test_J, nb_nodes)

	import scipy.sparse
	if dataset == 'KS20':
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([3, 2, 20, 8, 8, 9, 10, 9, 11, 10, 4, 20, 4, 5, 5, 6, 6, 7, 1, 20, 1, 0, 16, 0,
		                12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15, 2, 20, 11, 23, 10, 24, 7, 21, 6, 22])
		j_pair_2 = np.array([2, 3, 8, 20, 9, 8, 9, 10, 10, 11, 20, 4, 5, 4, 6, 5, 7, 6, 20, 1, 0, 1, 0, 16,
		                0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14, 20, 2, 23, 11, 24, 10, 21, 7, 22, 6])
		con_matrix = np.ones([48])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([25, 25])
	elif dataset == 'CASIA_B':
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13])
		j_pair_2 = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 11, 1, 12, 11, 13, 12])
		con_matrix = np.ones([26])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([14, 14])
		# adj_interp = generate_denser_adj(adj_joint)
	else:
		# Joint-Level adjacent matrix
		j_pair_1 = np.array([3, 2, 2, 8, 8, 9, 10, 9, 11, 10, 4, 2, 4, 5, 5, 6, 6, 7, 1, 2, 1, 0, 16, 0,
		                12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15])
		j_pair_2 = np.array([2, 3, 8, 2, 9, 8, 9, 10, 10, 11, 2, 4, 5, 4, 6, 5, 7, 6, 2, 1, 0, 1, 0, 16,
		                0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14])
		con_matrix = np.ones([38])
		adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
		if global_att:
			adj_joint = np.ones([20, 20])

	# Part-Level adjacent matrix
	p_pair_1 = np.array([5, 6, 5, 8, 6, 7, 8, 9, 5, 4, 4, 2, 4, 0, 2, 3, 1, 0])
	p_pair_2 = np.array([6, 5, 8, 5, 7, 6, 9, 8, 4, 5, 2, 4, 0, 4, 3, 2, 0, 1])
	con_matrix = np.ones([18])
	adj_part = scipy.sparse.coo_matrix((con_matrix, (p_pair_1, p_pair_2)), shape=(10, 10)).toarray()

	# Body-Level adjacent matrix
	b_pair_1 = np.array([2, 3, 2, 4, 2, 1, 2, 0])
	b_pair_2 = np.array([3, 2, 4, 2, 1, 2, 0, 2])
	con_matrix = np.ones([8])
	adj_body = scipy.sparse.coo_matrix((con_matrix, (b_pair_1, b_pair_2)), shape=(5, 5)).toarray()

	if global_att:
		adj_part = np.ones([10, 10])
		adj_body = np.ones([5, 5])

	if dataset == 'IAS':
		nb_classes = 11
	elif dataset == 'KGBD':
		nb_classes = 164
	elif dataset == 'BIWI':
		nb_classes = 28
	elif dataset == 'KS20':
		nb_classes = 20
	elif dataset == 'CASIA_B':
		nb_classes = 124

	adj_joint = adj_joint[np.newaxis]
	biases_joint = adj_to_bias(adj_joint, [nb_nodes], nhood=nhood)

	adj_part = adj_part[np.newaxis]
	biases_part = adj_to_bias(adj_part, [10], nhood=1)

	adj_body = adj_body[np.newaxis]
	biases_body = adj_to_bias(adj_body, [5], nhood=1)

	return X_train_J, X_train_P, X_train_B, y_train, X_test_J, X_test_P, X_test_B, y_test, \
	       adj_joint, biases_joint, adj_part, biases_part, adj_body, biases_body, nb_classes

"""
 Generate part-level  skeleton graphs.
"""

def reduce2part(X, joint_num=20):
	if joint_num == 25:
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head = [2, 3, 20]
		left_arm_up = [4, 5]
		left_arm_down = [6, 7, 21, 22]
		right_arm_up = [8, 9]
		right_arm_down = [10, 11, 23, 24]
	elif joint_num == 20:
		left_leg_up = [12, 13]
		left_leg_down = [14, 15]
		right_leg_up = [16, 17]
		right_leg_down = [18, 19]
		torso = [0, 1]
		head = [2, 3]
		left_arm_up = [4, 5]
		left_arm_down = [6, 7]
		right_arm_up = [8, 9]
		right_arm_down = [10, 11]
	elif joint_num == 14:
		left_leg_up = [11]
		left_leg_down = [12, 13]
		right_leg_up = [8]
		right_leg_down = [9, 10]
		torso = [1]
		head = [0]
		left_arm_up = [5]
		left_arm_down = [6, 7]
		right_arm_up = [2]
		right_arm_down = [3, 4]

	x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
	x_leftlegup = np.mean(X[:, :, left_leg_up, :], axis=2)
	x_leftlegdown = np.mean(X[:, :, left_leg_down, :], axis=2)
	x_rightlegup = np.mean(X[:, :, right_leg_up, :], axis=2)
	x_rightlegdown = np.mean(X[:, :, right_leg_down, :], axis=2)
	x_head = np.mean(X[:, :, head, :], axis=2)
	x_leftarmup = np.mean(X[:, :, left_arm_up, :], axis=2)
	x_leftarmdown = np.mean(X[:, :, left_arm_down, :], axis=2)
	x_rightarmup = np.mean(X[:, :, right_arm_up, :], axis=2)
	x_rightarmdown = np.mean(X[:, :, right_arm_down, :], axis=2)
	X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup,
	                         x_leftarmdown, x_rightarmup, x_rightarmdown), axis=-1) \
		.reshape([X.shape[0], X.shape[1], 10, 3])
	return X_part

"""
 Generate body-level  skeleton graphs.
"""

def reduce2body(X, joint_num=20):
	if joint_num == 25:
		left_leg = [12, 13, 14, 15]
		right_leg = [16, 17, 18, 19]
		torso = [0, 1, 2, 3, 20]
		left_arm = [4, 5, 6, 7, 21, 22]
		right_arm = [8, 9, 10, 11, 23, 24]
	elif joint_num == 20:
		left_leg = [12, 13, 14, 15]
		right_leg = [16, 17, 18, 19]
		torso = [0, 1, 2, 3]
		left_arm = [4, 5, 6, 7]
		right_arm = [8, 9, 10, 11]
	elif joint_num == 14:
		left_leg = [11, 12, 13]
		right_leg = [8, 9, 10]
		torso = [0, 1]
		left_arm = [5, 6, 7]
		right_arm = [2, 3, 4]

	x_torso = np.mean(X[:, :, torso, :], axis=2)  # [N * T, V=1]
	x_leftleg = np.mean(X[:, :, left_leg, :], axis=2)
	x_rightleg = np.mean(X[:, :, right_leg, :], axis=2)
	x_leftarm = np.mean(X[:, :, left_arm, :], axis=2)
	x_rightarm = np.mean(X[:, :, right_arm, :], axis=2)
	X_body = np.concatenate((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), axis=-1)\
		.reshape([X.shape[0], X.shape[1], 5, 3])
	return X_body

"""
 Calculate normalized area under curves.
"""
def cal_nAUC(scores, labels):
	scores = np.array(scores)
	labels = np.array(labels)
	# Compute micro-average ROC curve and ROC area
	fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())
	roc_auc = auc(fpr, tpr)
	return roc_auc

"""
 Generate training data with evenly distributed classes.
"""
def class_samp_gen(X, y, ids_, batch_size):
	class_num = len(ids_.keys())
	ids_ = sorted(ids_.items(), key=lambda item: item[0])
	cnt = 0
	all_batch_X = []
	all_batch_y = []
	total = y.shape[0]
	batch_num = total // batch_size * 2
	batch_num = total // batch_size * 2
	class_in_bacth = class_num
	batch_per_class = batch_size // class_in_bacth
	class_cnt = class_in_bacth
	# print(total, batch_num, batch_per_class)
	for i in range(batch_num):
		batch_X = []
		batch_y = []
		for k, v in ids_[class_cnt-class_in_bacth:class_cnt]:
			# print(k, len(v))
			# cnt += len(v)
			if len(v[batch_per_class*i:batch_per_class*(i+1)]) < batch_per_class:
				rand_ind = np.random.choice(len(v), batch_per_class)
				v_array = np.array(v)
				samp_per_class = v_array[rand_ind].tolist()
				batch_X.extend(samp_per_class)
			else:
				batch_X.extend(v[batch_per_class*i:batch_per_class*(i+1)])
			batch_y.extend(batch_per_class * [k])
		if class_cnt + class_in_bacth > class_num and class_cnt <= class_num:
			class_cnt = class_num
		else:
			class_cnt = class_cnt + class_in_bacth
		all_batch_X.extend(batch_X)
		all_batch_y.extend(batch_y)
	# print(len(all_batch_X), len(all_batch_y))
	X_train = X[all_batch_X]
	y_train = np.array(all_batch_y)
	return X_train, y_train

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
	nb_graphs = adj.shape[0]
	mt = np.empty(adj.shape)
	for g in range(nb_graphs):
		mt[g] = np.eye(adj.shape[1])
		for _ in range(nhood):
			mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
		for i in range(sizes[g]):
			for j in range(sizes[g]):
				if mt[g][i][j] > 0.0:
					mt[g][i][j] = 1.0
	return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
	"""Parse index file."""
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index


def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
	"""Load data."""
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
		tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range - min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range - min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	idx_val = range(len(y), len(y) + 500)

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	# print(adj.shape)
	# print(features.shape)

	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_random_data(size):
	adj = sp.random(size, size, density=0.002)  # density similar to cora
	features = sp.random(size, 1000, density=0.015)
	int_labels = np.random.randint(7, size=(size))
	labels = np.zeros((size, 7))  # Nx7
	labels[np.arange(size), int_labels] = 1

	train_mask = np.zeros((size,)).astype(bool)
	train_mask[np.arange(size)[0:int(size / 2)]] = 1

	val_mask = np.zeros((size,)).astype(bool)
	val_mask[np.arange(size)[int(size / 2):]] = 1

	test_mask = np.zeros((size,)).astype(bool)
	test_mask[np.arange(size)[int(size / 2):]] = 1

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	# sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation."""

	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx


def standardize_data(f, train_mask):
	"""Standardize feature matrix and convert to tuple representation"""
	# standardize data
	f = f.todense()
	mu = f[train_mask == True, :].mean(axis=0)
	sigma = f[train_mask == True, :].std(axis=0)
	f = f[:, np.squeeze(np.array(sigma > 0))]
	mu = f[train_mask == True, :].mean(axis=0)
	sigma = f[train_mask == True, :].std(axis=0)
	f = (f - mu) / sigma
	return f


def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
	num_nodes = adj.shape[0]
	adj = adj + sp.eye(num_nodes)  # self-loop
	adj[adj > 0.0] = 1.0
	if not sp.isspmatrix_coo(adj):
		adj = adj.tocoo()
	adj = adj.astype(np.float32)
	indices = np.vstack(
		(adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
	# return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
	return indices, adj.data, adj.shape
