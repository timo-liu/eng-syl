from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation, GRU,concatenate, ZeroPadding1D, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import pickle
import numpy as np
import os

class onc_to_phon:
    def __init__(self,e2i ='op_e2i.pkl', d2i= 'op_d2i.pkl', input_size=19, latent_dim=128, embed_dim=500):
        self.this_dir, this_filename = os.path.split(__file__)
        path_e2i = os.path.join(self.this_dir, e2i)
        path_d2i = os.path.join(self.this_dir, d2i)
        internal_d2i = pickle.load(open(path_d2i, 'rb'))
        self.i2d = {value: key for key, value in internal_d2i.items()}
        self.e2i = pickle.load(open(path_e2i, 'rb'))
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat_e = len(self.e2i) + 1
        self.max_feat_d = len(self.internal_d2i) + 1
        self.model = self.build_model()
        self.model.load_weights(os.path.join(self.this_dir,'op_best_weights.h5'))

    def build_model(self):

        # orthographic and ipa input layers
        ortho_inputs = Input(self.input_size, )
        # first branch ortho
        x = Embedding(self.max_feat_e, self.embed_dim, input_length=self.input_size)(ortho_inputs)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout=0.2,
                              activity_regularizer=regularizers.l2(1e-5)), input_shape=(self.input_size, 1))(x)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout=0.2,
                              activity_regularizer=regularizers.l2(1e-5)), input_shape=(self.input_size, 1))(x)
        x = ZeroPadding1D(padding=(0, self.input_size - x.shape[1]))(x)
        z = TimeDistributed(Dense(self.max_feat_d))(x)
        z = Activation('softmax')(z)

        model = Model(inputs=[ortho_inputs], outputs=z)

        return model

    def ipafy(self, word):
        inted_ortho = []
        for c in word:
            inted_ortho += [self.e2i[c]]

        inted_ortho = pad_sequences([inted_ortho], maxlen=self.input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.input_size, 1), verbose=0)[0]
        indexes = self.to_ind(predicted)
        converted = [self.i2d[x] for x in indexes if x != 0]
        return converted

    def to_ind(self, sequence):
        index_sequence = []
        for ind in sequence:
            index_sequence += [np.argmax(ind)]
        return index_sequence