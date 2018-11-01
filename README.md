### Deep learning projects

#### Project: toxic comment classification challenge Kaggle
1. toxic_comment_classification_CNN_Keras.ipynb
- Approach: CNN with 3 convolutional layers, 3 pooling layers and 1 dense layer
- Result: (10 epochs) training data: loss = 0.0503, accuracy = 0.9820; validation set: loss = 0.0898, accuracy = 0.9718

2. toxic_comment_classification_LSTM_Keras.ipynb
- Approach: 1 LSTM layer, 1 pooling layer, 1 dense layer
- Result: (2 epochs) training data: loss = 0.0573, accuracy = 0.9800; validation set: loss = 0.0576, accuracy = 0.9800

3. toxic_comment_classification_BidirectionalLSTM_Keras.ipynb
- Approach: 1 Bidirectional-LSTM layer, 1 pooling layer, 1 dense layer
- Result: (2 epochs) training data: loss = 0.0550, accuracy = 0.9805; validation set: loss = 0.0551, accuracy = 0.9803

#### Project: MNIST (digit recognizer)
1. MNSIT_Bidirectional-LSTM_Keras.ipynb
- Approach: 1 permute-dimension layer, 1 Bidirectional-LSTM layer, 1 pooling layer, 1 dense layer
- Result: (5 epochs) training data: loss = 0.1219, accuracy = 0.9646; validation set: loss = 0.1239, accuracy = 0.9627 

#### Project: Neural machine translation (English-to-Spanish)
1. neural-machine-translation_seq2seq_Keras.ipynb
- Approach: regular sequence-to-sequence modeling with encoder-decoder architecture
    - LSTM for encoder and decoder
    - teacher forcing

2. neural-machine-translation_seq2seq_attention_Keras.ipynb
- Approach:  sequence-to-sequence modeling with encoder-decoder architecture and attention
    - Bidirectional-LSTM for encoder
    - LSTM for decoder
    - Attention with 2 dense layers
    - teacher forcing
- Conclusion: Compared to regular seq-to-seq model, adding "attention" will increase the translation accuracy
    - Utilize all encoder's hidden states instead the last one
    - For each output word, "attention" tells the model which part of input sequence to be paid attention to

#### Project: bABI automatic text understanding and reasoning
1. bABI_memory_network_Keras.ipynb
- Approach: memory network for two supporting facts
- Conclusion: With 10000 training samples, and 1000 testing samples, the model ran very fast for 30 epochs with high accuracy. Training data: loss = 0.1848, accuracy = 0.9438; testing data: loss = 0.3510, accuracy = 0.8900

#### Project: word embedding
1. word2vec_tensorflow.py
2. glove_tensorflow.py

#### others
1. RNN_brown_dataset.ipynb
- word to index and word embedding
- sequence to sequence prediction using LSTM/GRU on brown corpus from NLTK tookit
- test trained word embedding with words similarity

2. poetry_classifier.ipynb
- classify poetry using POS tags sequence

3. RRNN_poetry_generating.ipynb
- simple Rated RNN for poetry generating (sequence to sequence prediction)

