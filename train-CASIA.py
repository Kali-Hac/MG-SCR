import time
import numpy as np
import tensorflow as tf
import os, sys
from models import GAT
from utils import process_reb as process
from sklearn.preprocessing import label_binarize

dataset = ''
split = ''      # dataset splits, only for IAS-A (A) and IAS-B (B)
pretext = ''    # pretext task: SSP
view_dir = ''
save_dirs = 'trained_models' # save_directory
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
pre_epochs = 150    # pretrain epochs
sample_num = 1  # number for sparse sampling
nb_nodes = 20   # number of nodes in the joint-level graph
nhood = 1   # n-hood = 1 -> neighbor nodes
coarse_lambda = 1   # fusion coefficient
ft_size = 3     # initial feature dimension (3D)
time_step = 6   # sequence length
LSTM_embed = 128    # number of hidden units in LSTM per layer
num_layers = 2  # number of LSTM layers

batch_size = 256
nb_epochs = 10000   # epochs for training (we set to 10000 with an early stopping strategy)
patience = 100  # early stopping paramters
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8]  # numbers of hidden units per each structural relation (attention) head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT # multi-head structural relation layer: 1-hop multi-head attention layer
tf.app.flags.DEFINE_string('dataset', 'CASIA_B', "Dataset: BIWI, IAS, KS20 or KGBD")
tf.app.flags.DEFINE_string('length', '20', "4, 6, 8 or 10")
tf.app.flags.DEFINE_string('split', '', "for IAS-Lab testing splits (A or B)")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('c_lambda', '0.3', "fusion coefficient")
tf.app.flags.DEFINE_string('task', 'pre', "Pre-training task: SSP or none")
tf.app.flags.DEFINE_string('view', '', "data splits for different views")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")
FLAGS = tf.app.flags.FLAGS

# check parameters
# if FLAGS.dataset not in ['BIWI', 'IAS', 'KGBD', 'KS20']:
# 	raise Exception('Dataset must be BIWI, IAS, KGBD, or KS20.')
if not FLAGS.gpu.isdigit() or int(FLAGS.gpu) < 0:
	raise Exception('GPU number must be a positive integer.')
# if FLAGS.length not in ['4', '6', '8', '10']:
# 	raise Exception('Length number must be 4, 6, 8 or 10.')
if FLAGS.split not in ['', 'A', 'B']:
	raise Exception('Datset split must be "A" (for IAS-A), "B" (for IAS-B), "" (for other datasets).')
if float(FLAGS.c_lambda) < 0 or float(FLAGS.c_lambda) > 1:
	raise Exception('Fusion coefficient must be not less than 0 or not larger than 1.')
if FLAGS.task not in ['pre', 'none']:
	raise Exception('Pre-training task must be "none" (no pre-training) or "pre" (SSP).')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset
time_step = int(FLAGS.length)
k_values = list(range(1, time_step))
levels = len(k_values)
coarse_lambda = float(FLAGS.c_lambda)
split = FLAGS.split
pretext = FLAGS.task

try:
	os.mkdir(save_dirs)
except:
	pass
if pretext == 'none':  # no pre-training
	pre_epochs = 0
if dataset == 'KS20':
	pre_epochs = 300
	nb_nodes = 25
elif dataset == 'CASIA_B':
	batch_size = 128
	LSTM_embed = 256
	lr = 0.0005
	# pre_epochs = 160
	pre_epochs = 300
	if FLAGS.probe_type != '':
		pre_epochs = 160
	nb_nodes = 14
if FLAGS.view != '':
	view_dir = '_view_' + FLAGS.view
else:
	view_dir = ''
if FLAGS.probe_type != '':
	view_dir = '_CME'
	view_dir = '_CME_' + str(LSTM_embed) + '_e'

print('Dataset: ' + dataset + split)
print('----- Opt. hyperparams -----')
print('pre_train_epochs: ' + str(pre_epochs))
print('nhood: ' + str(nhood))
print('skeleton_joints: ' + str(nb_nodes))
print('seqence_length: ' + str(time_step))
print('multi_level: ' + str(levels))
print('pre-training task: ' + str(pretext))
print('fusion_lambda: ' + str(coarse_lambda))
print('batch_size: ' + str(batch_size))
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. structural relation heads: ' + str(n_heads))
print('nonlinearity: ' + str(nonlinearity))
print('LSTM_embed_num: ' + str(LSTM_embed))
print('LSTM_layer_num: ' + str(num_layers))

"""
 Obtain training and testing data in Joint-level, Part-level, and Body-level.
 Generate corresponding adjacent matrix and bias.
"""
if FLAGS.probe_type == '':
	X_train_J, X_train_P, X_train_B, y_train, X_test_J, X_test_P, X_test_B, y_test, \
	adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, nb_classes = \
	process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
	                       nb_nodes=nb_nodes, nhood=nhood, global_att=False, batch_size=batch_size, view=FLAGS.view)
else:
	from utils import process_cme as process
	X_train_J, X_train_P, X_train_B, y_train, X_test_J, X_test_P, X_test_B, y_test, \
	adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, nb_classes = \
	process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
	                       nb_nodes=nb_nodes, nhood=nhood, global_att=False, batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)

with tf.Graph().as_default():
	with tf.name_scope('Input'):
		lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_classes))
		J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size*time_step, nb_nodes, ft_size))
		P_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 10, ft_size))
		B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 5, ft_size))
		J_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
		P_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 10, 10))
		B_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5))
		attn_drop = tf.placeholder(dtype=tf.float32, shape=())
		ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
		is_train = tf.placeholder(dtype=tf.bool, shape=())
	"""
		Multi-head Structural Relation Layer (MSRL) and Cross-level Collaborative Relation Layer (CCRL)
	"""
	with tf.name_scope("Multi_Level"), tf.variable_scope("Multi_Level", reuse=tf.AUTO_REUSE):
		def MSRL(J_in, J_bias_in, nb_nodes):
			W_h = tf.Variable(tf.random_normal([3, hid_units[-1]]))
			b_h = tf.Variable(tf.zeros(shape=[hid_units[-1], ]))
			J_h = tf.reshape(J_in, [-1, ft_size])
			J_h = tf.matmul(J_h, W_h) + b_h
			J_h = tf.reshape(J_h, [batch_size*time_step, nb_nodes, hid_units[-1]])
			J_seq_ftr = model.inference(J_h, 0, nb_nodes, is_train,
			                         attn_drop, ffd_drop,
			                         bias_mat=J_bias_in,
			                         hid_units=hid_units, n_heads=n_heads,
			                         residual=residual, activation=nonlinearity, r_pool=True)
			return J_seq_ftr


		def CCRL_fusion(s1, s2, s1_num, s2_num, hid_in):
			att_w = tf.nn.softmax(tf.matmul(s2, tf.transpose(s1, [0, 2, 1])))
			# [1536, 10, 20, 1]
			att_w = tf.expand_dims(att_w, axis=-1)
			# [1536, 1, 20, 8]
			s1 = tf.reshape(s1, [s1.shape[0], 1, s1.shape[1], hid_in])
			c_ftr = tf.reduce_sum(att_w * s1, axis=2)
			c_ftr = tf.reshape(c_ftr, [-1, hid_in])
			return c_ftr

		def part_info_back(part):
			if dataset == 'KS20':
				# left_leg_up = [12, 13]
				# left_leg_down = [14, 15]
				# right_leg_up = [16, 17]
				# right_leg_down = [18, 19]
				# torso = [0, 1]
				# head = [2, 3, 20]
				# left_arm_up = [4, 5]
				# left_arm_down = [6, 7, 21, 22]
				# right_arm_up = [8, 9]
				# right_arm_down = [10, 11, 23, 24]
				X = tf.tile(part[:, 4, :], [1, 2])
				X = tf.concat([X, tf.tile(part[:, 5, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 6, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 7, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 8, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 9, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 0, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 1, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 2, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 3, :], [1, 2])], axis=-1)

				X = tf.concat([X, tf.tile(part[:, 5, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 7, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 9, :], [1, 2])], axis=-1)

			elif dataset == 'CASIA_B':
				# left_leg_up = [11]
				# left_leg_down = [12, 13]
				# right_leg_up = [8]
				# right_leg_down = [9, 10]
				# torso = [1]
				# head = [0]
				# left_arm_up = [5]
				# left_arm_down = [6, 7]
				# right_arm_up = [2]
				# right_arm_down = [3, 4]

				X = tf.tile(part[:, 5, :], [1, 1])
				X = tf.concat([X, tf.tile(part[:, 4, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 8, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 9, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 6, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 7, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 2, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 3, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 0, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 1, :], [1, 2])], axis=-1)

			else:
				# left_leg_up = [12, 13]
				# left_leg_down = [14, 15]
				# right_leg_up = [16, 17]
				# right_leg_down = [18, 19]
				# torso = [0, 1]
				# head = [2, 3]
				# left_arm_up = [4, 5]
				# left_arm_down = [6, 7]
				# right_arm_up = [8, 9]
				# right_arm_down = [10, 11]
				X = tf.tile(part[:, 4, :], [1, 2])
				X = tf.concat([X, tf.tile(part[:, 5, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 6, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 7, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 8, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 9, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 0, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 1, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 2, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(part[:, 3, :], [1, 2])], axis=-1)

			X = tf.reshape(X, [batch_size*time_step, nb_nodes, -1])
			return X


		def body_info_back(body):
			if dataset == 'KS20':
				# left_leg = [12, 13, 14, 15]
				# right_leg = [16, 17, 18, 19]
				# torso = [0, 1, 2, 3, 20]
				# left_arm = [4, 5, 6, 7, 21, 22]
				# right_arm = [8, 9, 10, 11, 23, 24]
				X = tf.tile(body[:, 2, :], [1, 4])
				X = tf.concat([X, tf.tile(body[:, 3, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 4, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 0, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 1, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 2, :], [1, 1])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 3, :], [1, 2])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 4, :], [1, 2])], axis=-1)
			elif dataset == 'CASIA_B':
				# left_leg = [11, 12, 13]
				# right_leg = [8, 9, 10]
				# torso = [0, 1]
				# left_arm = [5, 6, 7]
				# right_arm = [2, 3, 4]
				X = tf.tile(body[:, 2, :], [1, 2])
				X = tf.concat([X, tf.tile(body[:, 4, :], [1, 3])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 3, :], [1, 3])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 1, :], [1, 3])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 0, :], [1, 3])], axis=-1)
			else:
				# left_leg = [12, 13, 14, 15]
				# right_leg = [16, 17, 18, 19]
				# torso = [0, 1, 2, 3]
				# left_arm = [4, 5, 6, 7]
				# right_arm = [8, 9, 10, 11]
				X = tf.tile(body[:, 2, :], [1, 4])
				X = tf.concat([X, tf.tile(body[:, 3, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 4, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 0, :], [1, 4])], axis=-1)
				X = tf.concat([X, tf.tile(body[:, 1, :], [1, 4])], axis=-1)
			X = tf.reshape(X, [batch_size * time_step, nb_nodes, -1])
			return X

		def CCRL(J_in, P_in, B_in, J_bias_in, P_bias_in, B_bias_in, hid_in, hid_out):
			h_J_seq_ftr = MSRL(J_in=J_in, J_bias_in=J_bias_in, nb_nodes=nb_nodes)
			h_P_seq_ftr = MSRL(J_in=P_in, J_bias_in=P_bias_in, nb_nodes=10)
			h_B_seq_ftr = MSRL(J_in=B_in, J_bias_in=B_bias_in, nb_nodes=5)

			h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_in])
			h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
			h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])

			W_cs_12 = tf.Variable(tf.random_normal([hid_in, hid_out]))
			W_cs_23 = tf.Variable(tf.random_normal([hid_in, hid_out]))
			W_cs_13 = tf.Variable(tf.random_normal([hid_in, hid_out]))

			c_12 = CCRL_fusion(h_J_seq_ftr, h_P_seq_ftr, nb_nodes, 10, hid_in)
			# r_12 = CCRL_fusion(h_P_seq_ftr, h_J_seq_ftr, 10, nb_nodes, hid_in)
			c_23 = CCRL_fusion(h_P_seq_ftr, h_B_seq_ftr, 10, 5, hid_in)
			# r_23 = CCRL_fusion(h_B_seq_ftr, h_P_seq_ftr, 5, 10, hid_in)
			c_13 = CCRL_fusion(h_J_seq_ftr, h_B_seq_ftr, nb_nodes, 5, hid_in)


			h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_in])
			h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
			h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])

			h_P_seq_ftr = h_P_seq_ftr + tf.matmul(c_12, W_cs_12)
			h_B_seq_ftr = h_B_seq_ftr + tf.matmul(c_23, W_cs_23)

			h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes,  hid_out])
			h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10,  hid_out])
			h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5,  hid_out])

			return h_B_seq_ftr, h_P_seq_ftr, h_J_seq_ftr

		h_B_seq_ftr, h_P_seq_ftr, h_J_seq_ftr = CCRL(J_in, P_in, B_in, J_bias_in, P_bias_in, B_bias_in,
	                                              hid_units[-1], hid_units[-1])
		x_s21 = part_info_back(h_P_seq_ftr)
		x_s31 = body_info_back(h_B_seq_ftr)
		seq_ftr = h_J_seq_ftr + coarse_lambda * (x_s21 + x_s31)
		seq_ftr = tf.reshape(seq_ftr, [-1, hid_units[-1]])

	ftr_in = J_in
	seq_ftr = tf.reshape(seq_ftr, [batch_size, time_step, -1])
	cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(LSTM_embed) for _ in range(num_layers)])

	# Sample feature sequences of different lengths: T_m_ftr
	# Generate target sequences to predict: Pred_tar
	# Generate consecutive sequences to test: Test_ftr
	T_m_ftr = []
	Pred_tar = []
	Test_ftr = []
	skes_in = tf.reshape(ftr_in, [batch_size, time_step, -1])
	# sample_num = 1
	for i in range(sample_num):
		for k in k_values:  # sparse sampling scheme
			T_m = np.random.choice(time_step-1, size=[k], replace=False).reshape(-1, 1)
			T_m = np.sort(T_m, axis=0) # sort
			Pred_t = T_m + 1
			T_m = T_m.astype(dtype=np.int32)
			Pred_t = Pred_t.astype(dtype=np.int32)
			T_m = np.tile(T_m, [batch_size, 1]).reshape(-1, 1)
			Pred_t = np.tile(Pred_t, [batch_size, 1]).reshape(-1, 1)
			seq_ind = np.arange(batch_size).reshape(-1, 1)
			seq_ind = np.tile(seq_ind, [1, k]).reshape(-1, 1)
			T_m = np.hstack([seq_ind, T_m])
			Pred_t = np.hstack([seq_ind, Pred_t])
			sampled_seq_ftr = tf.gather_nd(seq_ftr, T_m)
			Pred_t_seq = tf.gather_nd(skes_in, Pred_t)
			sampled_seq_ftr = tf.reshape(sampled_seq_ftr, [batch_size, k, -1])
			Pred_t_seq = tf.reshape(Pred_t_seq, [batch_size, k, -1])
			T_m_ftr.append(sampled_seq_ftr)   # sorted random frames (graph representations)
			Pred_tar.append(Pred_t_seq)
			if i == 0:
				T_m_test = np.arange(k).reshape(-1, 1)
				T_m_test = T_m_test.astype(dtype=np.int32)
				T_m_test = np.tile(T_m_test, [batch_size, 1]).reshape(-1, 1)
				T_m_test = np.hstack([seq_ind, T_m_test])
				test_seq_ftr = tf.gather_nd(seq_ftr, T_m_test)
				test_seq_ftr = tf.reshape(test_seq_ftr, [batch_size, k, -1])
				Test_ftr.append(test_seq_ftr)


	with tf.name_scope("Prediction"), tf.variable_scope("Prediction", reuse=tf.AUTO_REUSE):
		# Prediction Layer
		W1_pred = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
		b1_pred = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
		W2_pred = tf.Variable(tf.random_normal([LSTM_embed, nb_nodes * 3]))
		b2_pred = tf.Variable(tf.zeros(shape=[nb_nodes * 3, ]))
		all_pred_loss = []
		en_outs = []
		en_outs_whole = []
		en_outs_test = []
		ske_en_outs = []
		for i in range(levels * sample_num):
			if i < levels:
				test_seq_ftr = Test_ftr[i]
				encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, test_seq_ftr, dtype=tf.float32)
				encoder_output = tf.reshape(encoder_output, [-1, LSTM_embed])
				# Skeleton-level Prediction
				encoder_output = tf.reshape(encoder_output, [batch_size, k_values[i % levels], -1])
				en_outs_test.append(encoder_output[:, -1, :])
			#
			sampled_seq_ftr = T_m_ftr[i]
			# encoder_output: encoded graph states (h) of sparsely sampled subsequences
			encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, sampled_seq_ftr, dtype=tf.float32)
			# [256, 1, 128] => [256, 128]
			encoder_output = tf.reshape(encoder_output, [-1, LSTM_embed])
			# Skeleton Prediction
			pred_embedding = tf.nn.relu(tf.matmul(encoder_output, W1_pred) + b1_pred)
			pred_skeleton = tf.matmul(pred_embedding, W2_pred) + b2_pred
			pred_skeleton = tf.reshape(pred_skeleton, [batch_size, k_values[i % levels], nb_nodes*3])
			pred_loss = tf.reduce_mean(tf.nn.l2_loss(pred_skeleton - Pred_tar[i]))
			all_pred_loss.append(pred_loss)
			encoder_output = tf.reshape(encoder_output, [batch_size, k_values[i % levels], -1])
			en_outs.append(encoder_output[:, -1, :])
			if i == sample_num * levels - 1:
				en_outs_whole = encoder_output
		pred_opt = tf.train.AdamOptimizer(learning_rate=lr)
		pred_train_op = pred_opt.minimize(tf.reduce_mean(all_pred_loss))

	with tf.name_scope("Recognition"), tf.variable_scope("Recognition", reuse=tf.AUTO_REUSE):
		en_to_loss = en_outs[0]
		for i in range(1, levels * sample_num):
			en_to_loss = tf.concat([en_to_loss, en_outs[i]], axis=0)

		en_to_pred = en_outs_test[0]
		for i in range(1, levels):
			en_to_pred = tf.concat([en_to_pred, en_outs_test[i]], axis=0) # encoded graph states (h) for S_{1:f}
		# print(en_to_pred)
		# exit()
		# Recognition Layer
		W_1 = tf.Variable(tf.random_normal([LSTM_embed, LSTM_embed]))
		b_1 = tf.Variable(tf.zeros(shape=[LSTM_embed, ]))
		W_2 = tf.Variable(tf.random_normal([LSTM_embed, nb_classes]))
		b_2 = tf.Variable(tf.zeros(shape=[nb_classes, ]))
		# predict labels for h1, h2, ..., hf
		logits = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2

		logits_pred = tf.matmul(tf.nn.relu(tf.matmul(en_to_pred, W_1) + b_1), W_2) + b_2
		log_resh = tf.reshape(logits, [-1, nb_classes])
		lab_resh = tf.reshape(lbl_in, [-1, nb_classes])

		aver_pred = logits[:batch_size]
		aver_final_pred = logits_pred[:batch_size]

		# Average ID Prediction
		for i in range(1, levels * sample_num):
			aver_pred += logits[batch_size*i:batch_size*(i+1)]
			aver_final_pred += logits_pred[batch_size * i:batch_size * (i + 1)]

		correct_pred = tf.equal(tf.argmax(aver_pred, -1), tf.argmax(lab_resh, -1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		correct_final_pred = tf.equal(tf.argmax(aver_final_pred, -1), tf.argmax(lab_resh, -1))
		accuracy_final = tf.reduce_mean(tf.cast(correct_final_pred, tf.float32))
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=aver_pred, labels=lab_resh))

		train_op = model.training(loss,lr, l2_coef)

	saver = tf.train.Saver()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	vlss_mn = np.inf
	vacc_mx = 0.0
	vnAUC_mx = 0.0
	curr_step = 0

	X_train = X_train_J
	X_test = X_test_J
	with tf.Session(config=config) as sess:
		sess.run(init_op)
		train_loss_avg = 0
		train_acc_avg = 0
		val_loss_avg = 0
		val_acc_avg = 0

		for epoch in range(pre_epochs):
			tr_step = 0
			tr_size = X_train.shape[0]
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
				X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				loss_rec, loss_attr, acc_attr, loss_pred = 0, 0, 0, 0
				if pretext == 'pre':
					_, loss_pred = sess.run([pred_train_op, pred_loss],
					                                                 feed_dict={
						                                                 J_in: X_input_J,
						                                                 P_in: X_input_P,
						                                                 B_in: X_input_B,
						                                                 J_bias_in: biases_J,
						                                                 P_bias_in: biases_P,
						                                                 B_bias_in: biases_B,
						                                                 lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
						                                                 is_train: True,
						                                                 attn_drop: 0.0, ffd_drop: 0.0})
				tr_step += 1
			print('Epoch %s / %s   Pre-Train Loss (Temp. Pre.): %.5f' % (epoch, pre_epochs, loss_pred))

		lr = 0.0005
		vacc_early_model = 0
		vnAUC_mx = 0
		for epoch in range(nb_epochs):
			tr_step = 0
			tr_size = X_train.shape[0]
			train_acc_avg_final = 0
			logits_all = []
			labels_all = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
				X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				_, loss_value_tr, acc_tr, acc_tr_final, logits, labels = sess.run([train_op, loss, accuracy, accuracy_final, aver_final_pred, lab_resh],
				                  feed_dict={
					                  J_in: X_input_J,
					                  P_in: X_input_P,
					                  B_in: X_input_B,
					                  J_bias_in: biases_J,
					                  P_bias_in: biases_P,
					                  B_bias_in: biases_B,
					                  lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
					                  is_train: True,
					                  attn_drop: 0.0, ffd_drop: 0.0})
				logits_all.extend(logits.tolist())
				labels_all.extend(labels.tolist())
				train_loss_avg += loss_value_tr
				train_acc_avg += acc_tr
				train_acc_avg_final += acc_tr_final
				tr_step += 1
			train_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))
			rank_acc = {}
			vl_step = 0
			vl_size = X_test.shape[0]
			val_acc_avg_final = 0
			logits_all = []
			labels_all = []
			loaded_graph = tf.get_default_graph()
			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break
				X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
				X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				loss_value_vl, acc_vl, acc_vl_final, logits, labels = sess.run([loss, accuracy, accuracy_final, aver_final_pred, lab_resh],
				                                 feed_dict={
					                                 J_in: X_input_J,
					                                 P_in: X_input_P,
					                                 B_in: X_input_B,
					                                 J_bias_in: biases_J,
					                                 P_bias_in: biases_P,
					                                 B_bias_in: biases_B,
					                                 lbl_in: y_test[vl_step * batch_size:(vl_step + 1) * batch_size],
					                                 # msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
					                                 is_train: False,
					                                 attn_drop: 0.0, ffd_drop: 0.0})
				y_input = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
				for i in range(y_input.shape[0]):
					for K in range(1, nb_classes + 1):
						if K not in rank_acc.keys():
							rank_acc[K] = 0
						t = np.argpartition(logits[i], -K)[-K:]
						if np.argmax(y_input[i]) in t:
							rank_acc[K] += 1
				logits_all.extend(logits.tolist())
				labels_all.extend(labels.tolist())
				val_loss_avg += loss_value_vl
				val_acc_avg += acc_vl
				val_acc_avg_final += acc_vl_final
				vl_step += 1
			for K in rank_acc.keys():
				rank_acc[K] /= (vl_step * batch_size)
				rank_acc[K] = round(rank_acc[K], 4)
			val_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))
			if val_acc_avg / vl_step > vacc_mx:
				vacc_mx = val_acc_avg / vl_step
				vnAUC_mx = val_nAUC
			print('Epoch %s   Training: loss = %.5f, acc = %.5f, nAUC = %.5f | Val: loss = %.5f, acc = %.5f, max = %.5f, nAUC = %.5f, max = %.5f' %
			      (epoch, train_loss_avg / tr_step, train_acc_avg_final / tr_step, train_nAUC,
			       val_loss_avg / vl_step, val_acc_avg_final / vl_step, vacc_mx, val_nAUC, vnAUC_mx))

			if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
				if val_acc_avg / vl_step >= vacc_mx:
					vacc_early_model = val_acc_avg / vl_step
					vlss_early_model = val_loss_avg / vl_step
					# vnAUC_mx = val_nAUC
					# checkpt_file = save_dirs + '/' + str(coarse_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
					#                '-' + dataset + split + '/' + \
					#                 str(round(vacc_early_model*100, 1)) + '_' + str(round(vnAUC_mx*100, 1)) + '.ckpt'
					# print(checkpt_file)
					# print('Update Max Accuracy: %s  Max nAUC: %s' % (str(round(vacc_early_model*100, 1)), str(round(vnAUC_mx*100, 1))))
					checkpt_file = save_dirs + '/' + str(coarse_lambda) + '-' + pretext + '-' + str(nhood) + '-' + str(time_step) + \
					               '-' + dataset + split + '-' + view_dir + '/' + \
					                'best_model.ckpt'
					saver.save(sess, checkpt_file)
				# vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
				vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
				curr_step = 0
			else:
				curr_step += 1
				if curr_step == patience:
					print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx, ', nAUC: ',  vnAUC_mx)
					from sklearn.metrics import roc_curve, auc, confusion_matrix
					y_true = np.argmax(np.array(labels_all), axis=-1)
					y_pred = np.argmax(np.array(logits_all), axis=-1)
					print('\n### Re-ID Confusion Matrix: ')
					print(confusion_matrix(y_true, y_pred))
					print('### Rank-N Accuracy: ')
					print(rank_acc)
					print('### nAUC: ', vnAUC_mx)
					print()
					print('Dataset: ' + dataset + split)
					print('----- Opt. hyperparams -----')
					print('pre_train_epochs: ' + str(pre_epochs))
					print('nhood: ' + str(nhood))
					print('skeleton_joints: ' + str(nb_nodes))
					print('seqence_length: ' + str(time_step))
					print('multi_level: ' + str(levels))
					print('pre-training task: ' + str(pretext))
					print('fusion_lambda: ' + str(coarse_lambda))
					print('batch_size: ' + str(batch_size))
					print('lr: ' + str(lr))
					print('l2_coef: ' + str(l2_coef))
					print('----- Archi. hyperparams -----')
					print('nb. layers: ' + str(len(hid_units)))
					print('nb. units per layer: ' + str(hid_units))
					print('nb. structural relation heads: ' + str(n_heads))
					print('nonlinearity: ' + str(nonlinearity))
					print('LSTM_embed_num: ' + str(LSTM_embed))
					print('LSTM_layer_num: ' + str(num_layers))
					break
			train_loss_avg = 0
			train_acc_avg = 0
			val_loss_avg = 0
			val_acc_avg = 0
		saver.restore(sess, checkpt_file)
		sess.close()


