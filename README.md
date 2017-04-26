
# Quora question pairs: training

## Import packages


```python
%matplotlib inline
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime, time, json
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
```

## Initialize global variables


```python
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 17

```

## Load the dataset, embedding matrix and word count


```python
q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
with open(NB_WORDS_DATA_FILE, 'r') as f:
    
    nb_words = json.load(f)['nb_words']
```

## Partition the dataset into train and test sets


```python
X = np.stack((q1_data, q2_data), axis=1)

print(q1_data.shape, q2_data.shape)
y = labels
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]
```

    (404290, 25) (404290, 25)
    (404290,)


## Define the model


```python
Q1 = Sequential()
Q1.add(Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False))
Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
Q2 = Sequential()
Q2.add(Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False))
Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
model = Sequential()
model.add(Merge([Q1, Q2]))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])
```

## Train the model, checkpointing weights with best validation accuracy


```python
print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = model.fit([Q1_train, Q2_train],
                    y_train,
                    nb_epoch=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
```

    Starting training at 2017-04-26 12:41:15.901224
    Train on 327474 samples, validate on 36387 samples
    Epoch 1/17
    638s - loss: 0.6218 - acc: 0.6780 - precision: 0.6545 - recall: 0.2920 - fbeta_score: 0.3878 - val_loss: 0.6183 - val_acc: 0.6873 - val_precision: 0.6785 - val_recall: 0.2837 - val_fbeta_score: 0.3854
    Epoch 2/17
    634s - loss: 0.6162 - acc: 0.6830 - precision: 0.6722 - recall: 0.2939 - fbeta_score: 0.3960 - val_loss: 0.6123 - val_acc: 0.6838 - val_precision: 0.7341 - val_recall: 0.2146 - val_fbeta_score: 0.3189
    Epoch 3/17
    632s - loss: 0.6112 - acc: 0.6893 - precision: 0.6888 - recall: 0.3010 - fbeta_score: 0.4070 - val_loss: 0.6076 - val_acc: 0.6929 - val_precision: 0.6624 - val_recall: 0.3378 - val_fbeta_score: 0.4336
    Epoch 4/17
    633s - loss: 0.6060 - acc: 0.6957 - precision: 0.7129 - recall: 0.3064 - fbeta_score: 0.4167 - val_loss: 0.6122 - val_acc: 0.6970 - val_precision: 0.7442 - val_recall: 0.2695 - val_fbeta_score: 0.3807
    Epoch 5/17
    628s - loss: 0.6006 - acc: 0.7013 - precision: 0.7242 - recall: 0.3139 - fbeta_score: 0.4264 - val_loss: 0.6171 - val_acc: 0.6943 - val_precision: 0.7483 - val_recall: 0.2505 - val_fbeta_score: 0.3613
    Epoch 6/17
    631s - loss: 0.5938 - acc: 0.7078 - precision: 0.7460 - recall: 0.3209 - fbeta_score: 0.4374 - val_loss: 0.6101 - val_acc: 0.6983 - val_precision: 0.7725 - val_recall: 0.2521 - val_fbeta_score: 0.3664
    Epoch 7/17
    631s - loss: 0.5905 - acc: 0.7103 - precision: 0.7545 - recall: 0.3222 - fbeta_score: 0.4398 - val_loss: 0.6196 - val_acc: 0.7006 - val_precision: 0.7397 - val_recall: 0.2823 - val_fbeta_score: 0.3946
    Epoch 8/17
    633s - loss: 0.5877 - acc: 0.7134 - precision: 0.7641 - recall: 0.3239 - fbeta_score: 0.4429 - val_loss: 0.6139 - val_acc: 0.7051 - val_precision: 0.7523 - val_recall: 0.3000 - val_fbeta_score: 0.4140
    Epoch 9/17
    628s - loss: 0.5846 - acc: 0.7157 - precision: 0.7752 - recall: 0.3232 - fbeta_score: 0.4437 - val_loss: 0.6602 - val_acc: 0.7032 - val_precision: 0.7470 - val_recall: 0.2905 - val_fbeta_score: 0.4035
    Epoch 10/17
    626s - loss: 0.5813 - acc: 0.7182 - precision: 0.7793 - recall: 0.3284 - fbeta_score: 0.4500 - val_loss: 0.6756 - val_acc: 0.7002 - val_precision: 0.7688 - val_recall: 0.2586 - val_fbeta_score: 0.3734
    Epoch 11/17
    630s - loss: 0.5785 - acc: 0.7204 - precision: 0.7880 - recall: 0.3314 - fbeta_score: 0.4546 - val_loss: 0.6584 - val_acc: 0.7063 - val_precision: 0.7859 - val_recall: 0.2745 - val_fbeta_score: 0.3925
    Epoch 12/17
    629s - loss: 0.5736 - acc: 0.7235 - precision: 0.8006 - recall: 0.3333 - fbeta_score: 0.4580 - val_loss: 0.6584 - val_acc: 0.7086 - val_precision: 0.7800 - val_recall: 0.2874 - val_fbeta_score: 0.4048
    Epoch 13/17
    636s - loss: 0.5700 - acc: 0.7257 - precision: 0.8104 - recall: 0.3346 - fbeta_score: 0.4607 - val_loss: 0.6132 - val_acc: 0.7130 - val_precision: 0.7905 - val_recall: 0.3015 - val_fbeta_score: 0.4214
    Epoch 14/17
    627s - loss: 0.5683 - acc: 0.7268 - precision: 0.8139 - recall: 0.3365 - fbeta_score: 0.4634 - val_loss: 0.6619 - val_acc: 0.7105 - val_precision: 0.7972 - val_recall: 0.2838 - val_fbeta_score: 0.4038
    Epoch 15/17
    624s - loss: 0.5656 - acc: 0.7290 - precision: 0.8197 - recall: 0.3391 - fbeta_score: 0.4669 - val_loss: 0.7110 - val_acc: 0.7079 - val_precision: 0.7694 - val_recall: 0.2916 - val_fbeta_score: 0.4086
    Epoch 16/17
    626s - loss: 0.5623 - acc: 0.7309 - precision: 0.8280 - recall: 0.3399 - fbeta_score: 0.4689 - val_loss: 0.7328 - val_acc: 0.7113 - val_precision: 0.7958 - val_recall: 0.2876 - val_fbeta_score: 0.4082
    Epoch 17/17



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-3bab5836456b> in <module>()
          7                     validation_split=VALIDATION_SPLIT,
          8                     verbose=2,
    ----> 9                     callbacks=callbacks)
         10 t1 = time.time()
         11 print("Training ended at", datetime.datetime.now())


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/keras/models.py in fit(self, x, y, batch_size, nb_epoch, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, **kwargs)
        670                               class_weight=class_weight,
        671                               sample_weight=sample_weight,
    --> 672                               initial_epoch=initial_epoch)
        673 
        674     def evaluate(self, x, y, batch_size=32, verbose=1,


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/keras/engine/training.py in fit(self, x, y, batch_size, nb_epoch, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch)
       1190                               val_f=val_f, val_ins=val_ins, shuffle=shuffle,
       1191                               callback_metrics=callback_metrics,
    -> 1192                               initial_epoch=initial_epoch)
       1193 
       1194     def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/keras/engine/training.py in _fit_loop(self, f, ins, out_labels, batch_size, nb_epoch, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch)
        890                 batch_logs['size'] = len(batch_ids)
        891                 callbacks.on_batch_begin(batch_index, batch_logs)
    --> 892                 outs = f(ins_batch)
        893                 if not isinstance(outs, list):
        894                     outs = [outs]


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py in __call__(self, inputs)
       1898         session = get_session()
       1899         updated = session.run(self.outputs + [self.updates_op],
    -> 1900                               feed_dict=feed_dict)
       1901         return updated[:len(self.outputs)]
       1902 


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
        764     try:
        765       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 766                          run_metadata_ptr)
        767       if run_metadata:
        768         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
        962     if final_fetches or final_targets:
        963       results = self._do_run(handle, final_targets, final_fetches,
    --> 964                              feed_dict_string, options, run_metadata)
        965     else:
        966       results = []


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1012     if handle is None:
       1013       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
    -> 1014                            target_list, options, run_metadata)
       1015     else:
       1016       return self._do_call(_prun_fn, self._session, handle, feed_dict,


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1019   def _do_call(self, fn, *args):
       1020     try:
    -> 1021       return fn(*args)
       1022     except errors.OpError as e:
       1023       message = compat.as_text(e.message)


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1001         return tf_session.TF_Run(session, options,
       1002                                  feed_dict, fetch_list, target_list,
    -> 1003                                  status, run_metadata)
       1004 
       1005     def _prun_fn(session, handle, feed_dict, fetch_list):


    KeyboardInterrupt: 


## Plot training and validation accuracy


```python
acc = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                    'training': history.history['acc'],
                    'validation': history.history['val_acc']})
ax = acc.ix[:,:].plot(x='epoch', figsize={5,8}, grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0,1.0]);
```

## Print best validation accuracy and epoch


```python
max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
print('Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc))
```

## Evaluate the model with best validation accuracy on the test partition


```python
model.load_weights(MODEL_WEIGHTS_FILE)
loss, accuracy, precision, recall, fbeta_score = model.evaluate([Q1_test, Q2_test], y_test)
print('')
print('loss      = {0:.4f}'.format(loss))
print('accuracy  = {0:.4f}'.format(accuracy))
print('precision = {0:.4f}'.format(precision))
print('recall    = {0:.4f}'.format(recall))
print('F         = {0:.4f}'.format(fbeta_score))
```

# Prediction

# Load Test Data


```python
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
model.load_weights(MODEL_WEIGHTS_FILE)
QUESTION_PAIRS_FILE = 'test.csv'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TESTING_DATA_FILE = 'q1_test.npy'
Q2_TESTING_DATA_FILE = 'q2_test.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300

```


```python
import csv
question1 = []
question2 = []
with open(QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
print('Question pairs: %d' % len(question1))
```

    Question pairs: 2345796



```python
from keras.preprocessing.text import Tokenizer
questions = question1 + question2
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index

print("Words in index: %d" % len(word_index))
```

    Words in index: 101312



```python
print("Processing", GLOVE_FILE)

embeddings_index = {}
with open(GLOVE_FILE, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))
```

    Processing glove.840B.300d.txt
    Word embeddings: 2196016



```python
nb_words = min(MAX_NB_WORDS, len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
```

    Null word embeddings: 31446



```python
from keras.preprocessing.sequence import pad_sequences
q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#labels = np.array(is_duplicate, dtype=int)
print('Shape of question1 data tensor:', q1_data.shape)
print('Shape of question2 data tensor:', q2_data.shape)
#print('Shape of label tensor:', labels.shape)
```

    Shape of question1 data tensor: (2345796, 25)
    Shape of question2 data tensor: (2345796, 25)



```python
np.save(open(Q1_TESTING_DATA_FILE, 'wb'), q1_data)
np.save(open(Q2_TESTING_DATA_FILE, 'wb'), q2_data)
#np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
#np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
#with open(NB_WORDS_DATA_FILE, 'w') as f:
#    json.dump({'nb_words': nb_words}, f)
```


```python
q1_data = np.load(open(Q1_TESTING_DATA_FILE, 'rb'))
q2_data = np.load(open(Q2_TESTING_DATA_FILE, 'rb'))
X = np.stack((q1_data, q2_data), axis=1)
Q1_test = X[:,0]
Q2_test = X[:,1]
```


```python
pred = model.predict([Q1_test, Q2_test])
```


```python
#print("Starting training at", datetime.datetime.now())
#t0 = time.time()
name = "competition1.csv"
#pred will equal model.predict, which is a numpy array
pred = np.load(open('prediction.npy', 'rb'))
print('pred shape: ', pred.shape)


def write(pred, name):
    f = open(name, 'w')
    f.write('test_id,is_duplicate\n')
    for i in range(len(pred)):
        label = pred[i]
        image_id = i
        f.write(str(image_id) + ',' + str(label[0]) + '\n')
    f.close()
write(pred, name)
```

    pred shape:  (2345796, 1)



```python
print(pred.shape)
```

    (2345796, 1)



```python
np.save('prediction', pred)
```


```python

```
