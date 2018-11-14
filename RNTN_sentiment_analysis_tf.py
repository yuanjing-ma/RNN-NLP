import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score

def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


class Tree:
	def __init__(self, word, label):
		self.left = None
		self.right = None
		self.word = word 
		self.label = label


current_idx = 0
def str2tree(s, word2idx):
	global current_idx
	label = int(s[1])
	if s[3] == '(':
		t = Tree(None, label)
		child_s = s[3:]
		t.left = str2tree(child_s, word2idx)
		i = 0
		depth = 0
		for c in s:
			i += 1 
			if c == '(':
				depth += 1
			elif c == ')':
				depth -= 1
			if depth == 1:
				break
		t.right = str2tree(s[i+1], word2idx)
		return t 
	else:
		r = s.split(')', 1)[0]
		word = r[3:].lower()
		if word not in word2idx:
			word2idx[word] = current_idx
			current_idx += 1 
		t = Tree(word2idx[word], label)
		return t 


def get_ptb_data():
	if not os.path.exists('../large_files/trees'):
		print("need to create folder ../large_files/trees")
		exit()
	elif not os.path.exists('../large_files/trees/train.txt'):
		print("need to download train.txt dataset and store under trees folder")
		exit()
	elif not os.path.exists('../large_files/trees/test.txt'):
		print("need to download test.txt dataset and store and trees folder")

	word2idx = {}
	train = []
	test = []
	for line in open('../large_files/trees/train.txt'):
		line = line.rstrip()
		if line:
			t = str2tree(line, word2idx)
			train.append(t)
	for line in open('../large_files/trees/test.txt'):
		line = line.rstrip()
		if line:
			t = str2tree(line, word2idx)
			test.append(t)
	return train, test, word2idx


class RNTN:
	def __init__(self, V, D, K, activation = tf.tanh):
		self.V = V 
		self.D = D 
		self.K = K 
		self.f = activation 

	def fit(self, trees, test_trees, reg = 1e-3, epochs = 8, train_inner_nodes = False):
		D = self.D
		V = self.V 
		K = self.K 
		N = len(trees)

		We = init_weight(V, D)
		W11 = np.random.randn(D, D, D) / np.sqrt(3*D)
		W22 = np.random.randn(D, D, D) / np.sqrt(3*D)
		W12 = np.random.randn(D, D, D) / np.sqrt(3*D)
		W1 = init_weight(D, D)
		W2 = init_weight(D, D)
		bh = np.zeros(D)
		Wo = init_weight(D, K)
		bo = np.zeros(K)

		self.We = tf.Variable(We.astype(np.float32))
		self.W11 = tf.Variable(W11.astype(np.float32))
		self.W22 = tf.Variable(W22.astype(np.float32))
		self.W12 = tf.Variable(W12.astype(np.float32))
		self.W1 = tf.Variable(W1.astype(np.float32))
		self.W2 = tf.Variable(W2.astype(np.float32))
		self.bh = tf.Variable(bh.astype(np.float32))
		self.Wo = tf.Variable(Wo.astype(np.float32))
		self.bo = tf.Variable(bo.astype(np.float32))
		self.weights = [self.We, self.W11, self.W22, self.W12, self.W1, self.W2, self.Wo]

		words = tf.placeholder(tf.int32, shape = (None,), name = 'words')
		left_children = tf.placeholder(tf.int32, shape = (None,), name = 'left_children')
		right_children = tf.placeholder(tf.int32, shape = (None,), name = 'right_children')
		labels = tf.placeholder(tf.int32, shape = (None,), names = 'labels')

		self.words = words
		self.left = left_children
		self.right = right_children
		self.labels = labels

		def dot1(a, B):
			return tf.tensordot(a, B, axes = [[0], [1]])

		def dot2(B, a):
			return tf.tensordot(B, a, axes = [[1], [0]])

		def recursive_net_transform(hiddens, n):
			h_left = hiddens.read(left_children[n])
			h_right = hiddens.read(right_children[n])
			return self.f(
				dot1(h_left, dot2(self.W11, h_left)) + 
				dot1(h_right, dot2(self.W22, h_right)) + 
				dot1(h_left, dot2(self.W12, h_right)) + 
				dot1(h_left, self.W1) + 
				dot1(h_right, self.W2) + 
				self.bh
				)

		def recurrence(hiddens, n):
			w = words[n]
			h_n = tf.cond(
				w >= 0,
				lambda: tf.nn.embedding_lookup(self.We, w),
				lambda: recursive_net_transform(hiddens, n)
				)
			hiddens = hiddens.write(n, h_n)
			n = tf.add(n, 1)
			return hiddens, n 

		def condition(hiddens, n):
			return tf.less(n, tf.shape(words)[0])

        hiddens = tf.TensorArray(
        	tf.float32,
        	size = 0,
        	dynamic_size = True,
        	clear_after_read = False,
        	infer_shape = False
        	)

		hiddens, _ = tf.while_loop(
			condition,
			recurrence,
			[hiddens, tf.constant(0)],
			parallel_iterations = 1
			)
		h = hiddens.stack()
		logits = tf.matmul(h, self.Wo) + self.bo

		prediction_op = tf.argmax(logits, axis = 1)
		self.prediction_op = prediction_op

		rcost = reg * sum(tf.nn.l2_loss(p) for p in self.weights)
		if train_inner_nodes:
			labeled_indices = tf.where(labels >= 0)
			cost_op = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits = tf.gather(logits, labeled_indices),
					labels = tf.gather(labels, labeled_indices),
					)
				) + rcost
		else:
			cost_op = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits = logits[-1],
					labels = labels[-1]
					)
				) + rcost

		train_op = tf.train.AdagradOptimizer(learning_rate = 1e-4).minimize(cost_op)

		self.session = tf.Session()
		init_op = tf.global_variables_initializer()
		self.session.run(init_op)

		costs = []
		sequence_indexes = range(N)
		for i in range(epochs):
			t0 = datetime.now()
			sequence_indexes = shuffle(sequence_indexes)
			n_correct = 0
			n_total = 0 
			cost = 0
			it = 0 
			for j in sequence_indexes:
				words_, left, right, lab = trees[j]
				c, p, _ = self.session.run(
					(cost_op, prediction_op, train_op),
					feed_dict = {words: words_, left_children: left, right_children: right, labels: lab}
					)
				cost += c 
				n_correct += (p[-1] == lab[-1])
				n_total += 1 

				it += 1 
				if it % 10 == 0:
					sys.stdout.write(
						"j/N: %d/%d correct rate so far: %f, cost so far: %f\r" % (it, N, float(n_correct)/n_total, cost)
						)
					sys.stdout.flush()

			n_test_correct = 0
			n_test_total = 0 
			for words_, left, right, lab in test_trees:
				p = self.session.run(prediction_op, feed_dict = {words: words_, left_children: left, right_children: right, labels: lab})
                n_test_correct += (p[-1] == lab[-1])
                n_test_total += 1

            costs.append(cost)
        
        plt.plot(costs)
        plt.show()

    def predict(self, words, left, right, lab):
    	return self.session.run(
    		self.prediction_op,
    		feed_dict = {self.words: words, self.left: left, self.right: right, self.labels: lab}
    		)

    def score(self, trees):
    	n_total = len(trees)
    	n_correct = 0 
    	for words, left, right, lab in trees:
    		p = self.predict(words, left, right, lab)
    		n_correct += (p[-1] == lab[-1])
    	return float(n_correct) / n_total 

    def f1_score(self, trees):
    	Y = []
    	P = []
    	for words, left, right, lab in trees:
    		p = self.predict(words, left, right, lab)
    		Y.append(lab[-1])
    		P.append(p[-1])
    	return f1_score(Y, P, average = None).mean()


def add_idx_to_tree(tree, current_idx):
	if tree is None:
		return current_idx 
	current_idx = add_idx_to_tree(tree.left, current_idx)
	current_idx = add_idx_to_tree(tree.right, current_idx)
	tree.idx = current_idx
	current_idx += 1 
	return current_idx 


def tree2list(tree, parent_idx, is_binary = False):
	if tree is None:
		return [], [], [], []

	words_left, left_child_left, right_child_left, labels_left = tree2list(tree.left, tree.idx, is_binary)
	words_right, left_child_right, right_child_right, labels_right = tree2list(tree.right, tree.idx, is_binary)
    
    if tree.word is None:
    	w = -1
    	left = tree.left.idx
    	right = tree.right.idx 
    else:
    	w = tree.word
    	left = -1 
    	right = -1
    
    words = words_left + words_right + [w]
    left_child = left_child_left + left_child_right + [left]
    right_child = right_child_left + right_child_right + [right]

    if is_binary:
    	if tree.label > 2:
    		label = 1 
    	elif tree.label < 2:
    		label = 0
    	else:
    		label = -1 
    else:
    	label = tree.label 
    labels = labels_left + labels_right + [label]

    return words, left_child, right_child, labels 


def main(is_binary = True):
	train, test, word2idx = get_ptb_data

	for t in train:
		add_idx_to_tree(t, 0)
	train = [tree2list(t, -1, is_binary) for t in train]
	if is_binary:
		train = [t for t in train if t[3][-1] >= 0]

	for t in test:
		add_idx_to_tree(t, 0)
	test = [tree2list(t, -1, is_binary) for t in test]
	if is_binary:
		test = [t for t in test if t[3][-1] >= 0]

	train = shuffle(train)
	test = shuffle(test)
	V = len(word2idx)
	D = 10 
	K = 2 if is_binary else 5 

	model = RNTN(V, D, K)
	model.fit(train, test, reg = 1e-3, epochs = 20, train_inner_nodes = True)

if __name__ == '__main__':
	main()
