import os
import json
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.utils import shuffle
import import_ipynb
from RNN_brown_dataset import get_sentences_with_word2idx_limit_vocab
from word2vec_tensorflow import analogy

class Glove:
	def __init__(self, D, V, context_sz):
		self.D = D
		self.V = V 
		self.context_sz = context_sz 

	def fit(self, sentences, cc_matrix = None, learning_rate = 1e-4, reg = 0.1, xmax = 100, alpha = 0.75, epochs = 10):
		V = self.V 
		D = self.D 

		if not os.path.exists(cc_matrix):
			X = np.zeros(V, V)
			N = len(sentences)
			it = 0
			for sentence in sentences:
				it += 1 
				if it % 10000 == 0:
					print("processed", it, "/", N)
				n = len(sentence)
				for i in range(n):
					wi = sentence[i]
					start = max(0, i - self.context_sz)
					end = min(n, i + self.context_sz)
					 
					# make sure "start" and "end" token are part of the context
					# word2idx: "START":0, "END":1
					if i - self.context_sz < 0:
						points = 1.0 / (i + 1)
						X[wi, 0] += points
						X[0, wi] += points
					if i + self.context_sz > n:
						points = 1.0 / (n-i)
						X[wi, 1] += points 
						X[1, wi] += points 

					for j in range(start, i):
						wj = sentence[j]
						points = 1.0 / (i-j)
						X[wi, wj] += points
						X[wj, wi] += poitns 

					for j in range(i + 1, end):
						wj = sentence[j]
						points = 1.0 / (j-i)
						X[wi, wj] += points
						X[wj, wi] += points 
			np.save(cc_matrix, X)
		
		else:
			X = np.load(cc_matrix)

		print("max in X:", X.max())

        fX = np.zeros(V, V)
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X > xmax] = 1

        logX = np.log(X + 1)
        W = np.random.randn(V, D) / np.sqrt(V + D)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        c = np.zeros(V)
        mu = logX.mean()

        tfW = tf.Variable(W.astype(np.float32))
        tfU = tf.Variable(U.astype(np.float32))
        tfb = tf.Variable(b.reshape(V, 1).astype(np.float32))
        tfc = tf.Variable(c.reshape(1, V).astype(np.float32))
        tfLogX = tf.placeholder(tf.float32, shape = (V, V))
        tffX = tf.placeholder(tf.float32, shape = (V, V))

        delta = tf.matmul(tf.W, tf.transpose(tfU)) + tfb + tfc + mu - tfLogX
        cost = tf.reduce_mean(tffX * delta * delta)
        regularized_cost = cost 
        for param in (tfW, tfU):
        	regularized_cost += reg*tf.reduce_sum(param * param)

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(regularized_cost)
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)

        costs = []
        
        for epoch in range(epochs):
        	c, _ = session.run((cost, train_op), feed_dict = {tfLogX: logX, tffX: fX})
        	costs.append(c)

        self.W, self.V = session.run([tfW, tfU])
        plt.plot(costs)
        plt.show()

    def save(self, fn):
    	arrays = [self.W, self.U.T]
    	np.savez(fn, *arrays)


def main(we_file, w2i_file):
    cc_matrix = "cc_matrix_brown.npy"
    
    keep_words = set([
        'king', 'man', 'woman','france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
        'french', 'english', 'japan', 'japanese', 'chinese', 'italian','australia', 'australian', 
        'december', 'november', 'june', 'january', 'february', 'march', 'april', 'may', 'july', 
        'august', 'september', 'october',
        ])
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab = 5000, keep_words = keep_words)
    with open(w2i_file, 'w') as f:
    	json.dump(wor2idx, f)

    V = len(word2idx)
    model = Glove(100, V, 10)
    model.fit(sentences, cc_matrix = cc_matrix, epochs = 200)
    model.save(we_file)


if __name__ == '__main__':
	we = 'glove_model_brown.npz'
	w2i = 'glove_word2idx_brown.json'
	main(we, w2i)

	npz = np.load(we)
	W1 = npz['arr_0']
	W2 = npz['arr_1']

	with open(w2i) as f:
		word2idx = json.load(f)
		idx2word = {i:w for w, i in word2idx.items()}
    
    for concat in (True, False):
    	if concat:
    		We = np.hstack([W1, W2.T])
    	else We = (W1 + W2.T) / 2

    	analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
    	analogy('france', 'paris', 'england', 'london', word2idx, idx2word, We)
