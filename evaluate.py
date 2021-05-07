from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import tensorflow as tf
import os, sys
from models import GAT
from utils import process

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
nhood = 1
ft_size = 3
time_step = 6

batch_size = 256
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8]  # numbers of hidden units per each structural relation (attention) head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT # multi-head structural relation layer: 1-hop multi-head attention layer

tf.app.flags.DEFINE_string('dataset', 'BIWI', "Dataset: BIWI, IAS, KS20 or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")
tf.app.flags.DEFINE_string('split', '', "for IAS-Lab testing splits (A or B)")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('model_dir', 'best', "model directory")  # 'best' will test the best model in current directory


FLAGS = tf.app.flags.FLAGS

# check parameters
if FLAGS.dataset not in ['BIWI', 'IAS', 'KGBD', 'KS20']:
	raise Exception('Dataset must be BIWI, IAS, KGBD, or KS20.')
if not FLAGS.gpu.isdigit() or int(FLAGS.gpu) < 0:
	raise Exception('GPU number must be a positive integer.')
if FLAGS.length not in ['4', '6', '8', '10']:
	raise Exception('Length number must be 4, 6, 8 or 10.')
if FLAGS.split not in ['', 'A', 'B']:
	raise Exception('Datset split must be "A" (for IAS-A), "B" (for IAS-B), "" (for other datasets).')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset
time_step = int(FLAGS.length)
split = FLAGS.split
model_dir = FLAGS.model_dir


def evaluate_reid(model_dir, dataset):
	if dataset == 'BIWI':
		classes = list(range(28))
	elif dataset == 'IAS':
		classes = list(range(11))
	elif dataset == 'KS20':
		classes = list(range(20))
	elif  dataset == 'KGBD':
		classes = list(range(164))
	checkpoint = model_dir + ".ckpt"
	print('Evaluating the model saved in ' + model_dir)
	loaded_graph = tf.get_default_graph()

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpoint + '.meta')
		loader.restore(sess, checkpoint)
		lbl_in = loaded_graph.get_tensor_by_name('Input/Placeholder:0')
		J_in = loaded_graph.get_tensor_by_name('Input/Placeholder_1:0')
		P_in = loaded_graph.get_tensor_by_name('Input/Placeholder_2:0')
		B_in = loaded_graph.get_tensor_by_name('Input/Placeholder_3:0')
		J_bias_in = loaded_graph.get_tensor_by_name('Input/Placeholder_4:0')
		P_bias_in = loaded_graph.get_tensor_by_name('Input/Placeholder_5:0')
		B_bias_in = loaded_graph.get_tensor_by_name('Input/Placeholder_6:0')
		attn_drop = loaded_graph.get_tensor_by_name('Input/Placeholder_7:0')
		ffd_drop = loaded_graph.get_tensor_by_name('Input/Placeholder_8:0')
		is_train = loaded_graph.get_tensor_by_name('Input/Placeholder_9:0')
		aver_pre = loaded_graph.get_tensor_by_name('Recognition/Recognition/add_11:0')
		accuracy = loaded_graph.get_tensor_by_name('Recognition/Recognition/Mean:0')
		loss = loaded_graph.get_tensor_by_name('Recognition/Recognition/Mean_2:0')
		rank_acc = {}

		ts_size = X_test_J.shape[0]
		logits_all = []
		labels_all = []
		ts_step = 0
		ts_loss = 0.0
		ts_acc = 0.0

		while ts_step * batch_size < ts_size:
			if (ts_step + 1) * batch_size > ts_size:
				break
			X_input_J = X_test_J[ts_step * batch_size:(ts_step + 1) * batch_size]
			X_input_J = X_input_J.reshape([-1, nb_nodes, 3])
			X_input_P = X_test_P[ts_step * batch_size:(ts_step + 1) * batch_size]
			X_input_P = X_input_P.reshape([-1, 10, 3])
			X_input_B = X_test_B[ts_step * batch_size:(ts_step + 1) * batch_size]
			X_input_B = X_input_B.reshape([-1, 5, 3])
			y_input = y_test[ts_step * batch_size:(ts_step + 1) * batch_size]
			loss_value_ts, acc_ts, pred = sess.run([loss, accuracy, aver_pre],
			                                 feed_dict={
				                                 J_in: X_input_J,
				                                 P_in: X_input_P,
				                                 B_in: X_input_B,
				                                 J_bias_in: biases_J,
				                                 P_bias_in: biases_P,
				                                 B_bias_in: biases_B,
				                                 lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
				                                 is_train: False,
				                                 attn_drop: 0.0, ffd_drop: 0.0})
			for i in range(y_input.shape[0]):
				for K in range(1, len(classes) + 1):
					if K not in rank_acc.keys():
						rank_acc[K] = 0
					t = np.argpartition(pred[i], -K)[-K:]
					if np.argmax(y_input[i]) in t:
						rank_acc[K] += 1
			logits_all.extend(pred.tolist())
			labels_all.extend(y_input.tolist())
			ts_loss += loss_value_ts
			ts_acc += acc_ts
			ts_step += 1
		for K in rank_acc.keys():
			rank_acc[K] /= (ts_step * batch_size)
			rank_acc[K] = round(rank_acc[K], 4)
		val_nAUC = process.cal_nAUC(scores=np.array(logits_all), labels=np.array(labels_all))
		from sklearn.metrics import roc_curve, auc, confusion_matrix
		y_true = np.argmax(np.array(labels_all), axis=-1)
		y_pred = np.argmax(np.array(logits_all), axis=-1)
		print('\n### Re-ID Confusion Matrix: ')
		print(confusion_matrix(y_true, y_pred))
		print('### Rank-N Accuracy: ')
		print(rank_acc)
		print('### Test loss:', round(ts_loss / ts_step, 4), '; Test accuracy:', round(ts_acc / ts_step, 4),
		      '; Test nAUC:', round(val_nAUC, 4))
		exit()


if dataset == 'KS20':
	nb_nodes = 25
X_train_J, X_train_P, X_train_B, y_train, X_test_J, X_test_P, X_test_B, y_test, \
adj_J, biases_J, adj_P, biases_P, adj_B, biases_B, nb_classes = \
	process.gen_train_data(dataset=dataset, split=split, time_step=time_step,
	                       nb_nodes=nb_nodes, nhood=nhood, global_att=False, batch_size=batch_size)
if model_dir == 'best':
	if dataset == 'BIWI':
		model_dir = 'trained_models/best_models/BIWI_61.6_91.9'
	elif dataset == 'IAS' and split == 'A':
		model_dir = 'trained_models/best_models/IASA_56.5_87.0'
	elif dataset == 'IAS' and split == 'B':
		model_dir = 'trained_models/best_models/IASB_65.9_93.1'
	elif dataset == 'KS20':
		model_dir = 'trained_models/best_models/KS20_87.3_95.5'
	elif dataset == 'KGBD':
		model_dir = 'trained_models/best_models/KGBD_96.3_99.9'

evaluate_reid(model_dir, dataset)