### Deep learning projects

#### Dataset: toxic comment classification challenge Kaggle
1. toxic_comment_classification_CNN_Keras.ipynb
- Approach: CNN with 3 convolutional layers, 3 pooling layers and 1 dense layer
- Result: (10 epochs) training data: loss = 0.0503, accuracy = 0.9820; validation set: loss = 0.0898, accuracy = 0.9718

2. toxic_comment_classification_LSTM_Keras.ipynb
- Approach: 1 LSTM layer, 1 pooling layer, 1 dense layer
- Result: (2 epochs) training data: loss = 0.0573, accuracy = 0.9800; validation set: loss = 0.0576, accuracy = 0.9800

3. toxic_comment_classification_BidirectionalLSTM_Keras.ipynb
- Approach: 1 Bidirectional-LSTM layer, 1 pooling layer, 1 dense layer
- Result: (2 epochs) training data: loss = 0.0550, accuracy = 0.9805; validation set: loss = 0.0551, accuracy = 0.9803

#### Dataset: MNIST (digit recognizer)
1. MNSIT_Bidirectional-LSTM_Keras.ipynb
- Approach: 1 permute-dimension layer, 1 Bidirectional-LSTM layer, 1 pooling layer, 1 dense layer
- Result: (5 epochs) training data: loss = 0.1219, accuracy = 0.9646; validation set: loss = 0.1239, accuracy = 0.9627 
