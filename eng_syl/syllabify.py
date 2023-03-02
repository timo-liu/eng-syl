from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import pickle
import numpy as np


class Syllabel:
    def __init__(self, input_size=34, e2i=pickle.load(open('e2i.pkl', 'rb')), latent_dim=500, embed_dim=500, max_feat=61):
        self.e2i = e2i
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.max_feat = max_feat + 1  # include dim for padding value 0 (no corresponding index in dict)
        self.model = Sequential()
        self.model.add(Input(input_size, ))
        self.model.add(Embedding(self.max_feat, self.embed_dim, input_length=self.input_size))
        self.model.add(Bidirectional(LSTM(self.latent_dim, return_sequences=True, recurrent_dropout=0.4),
                                     input_shape=(input_size, 1)))
        self.model.add(TimeDistributed(Dense(3)))
        self.model.add(Activation('softmax'))

    def ignore_class_accuracy(self, to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy

        return ignore_accuracy

    def fit(self, x_tr, y_tr, x_test, y_test, ep, batch_size, save_filename):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy', self.ignore_class_accuracy(0)])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        ck = ModelCheckpoint(filepath=save_filename, monitor='accuracy', verbose=1, save_best_only=True,
                             mode='max')
        Callbacks = [es, ck]
        self.model.fit(x_tr, y_tr, epochs=ep, callbacks=Callbacks, batch_size=batch_size,
                       validation_data=(x_test, y_test))

    def syllabify(self, word):
        inted = []
        for c in word.lower():
            inted += [self.e2i[c]]
        inted = pad_sequences([inted], maxlen=self.input_size, padding='post')[0]
        predicted = self.model.predict(inted.reshape(1, self.input_size, 1, verbose=0))[0]
        converted = self.to_ind(predicted)

        return self.insert_syl(word, converted)

    def test_predict(self, x_tr, y_tr, ind):
        print(y_tr[ind])
        results = self.model.predict(x_tr[ind].reshape(1, self.input_size, 1))[0]
        print(self.to_ind(results))

    def to_ind(self, sequence):
        index_sequence = []
        for ind in sequence:
            index_sequence += [np.argmax(ind)]
        return index_sequence

    def insert_syl(self, word, indexes):
        index_list = np.where(np.array(indexes) == 2)[0]
        word_array = [*word]
        for i in range(0, len(index_list)):
            word_array.insert(index_list[i] + i + 1, '-')
        return ''.join(word_array)