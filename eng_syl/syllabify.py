from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation, GRU, ZeroPadding1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

class Syllable:
    def __init__(self, e2i_ortho='e2i.pkl', ortho_input_size=45, latent_dim=256, embed_dim=256, max_feat=259):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        path_clean = os.path.join(self.this_dir, 'clean.pkl')
        with open(path_clean, 'rb') as f:
            self.clean = pickle.load(f)
        path_e2i = os.path.join(self.this_dir, e2i_ortho)
        with open(path_e2i, 'rb') as f:
            self.e2i_ortho = pickle.load(f)
        self.ortho_input_size = ortho_input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat = max_feat
        self.model = self.build_model()
        self.model.load_weights(os.path.join(self.this_dir, 'syllabler_best_weights.h5'))

    def build_model(self):
        # orthographic input layer
        ortho_inputs = Input(shape=(self.ortho_input_size,))
        # embedding layer
        x = Embedding(input_dim=self.max_feat, output_dim=self.embed_dim)(ortho_inputs)
        # bidirectional GRU layers
        x = Bidirectional(GRU(units=self.latent_dim, return_sequences=True, recurrent_dropout=0.2, activity_regularizer=regularizers.l2(1e-5)))(x)
        x = Bidirectional(GRU(units=self.latent_dim, return_sequences=True, recurrent_dropout=0.2, activity_regularizer=regularizers.l2(1e-5)))(x)
        # time distributed dense layer
        x = TimeDistributed(Dense(units=3))(x)
        # activation layer
        z = Activation('softmax')(x)
        # define the model
        model = Model(inputs=[ortho_inputs], outputs=z)
        return model

    def syllabify(self, word):
        if word in self.clean:
            return self.clean[word]
        inted_ortho = [self.e2i_ortho[c] for c in word.lower()]
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size), verbose=0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(word, indexes)
        return converted

    def machine_syllabify(self, word):
        inted_ortho = [self.e2i_ortho[c] for c in word.lower()]
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size), verbose=0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(word, indexes)
        return converted

    def to_ind(self, sequence):
        index_sequence = [np.argmax(ind) for ind in sequence]
        return index_sequence

    def insert_syl(self, word, indexes):
        index_list = np.where(np.array(indexes) == 2)[0]
        word_array = list(word)
        for i in range(len(index_list)):
            word_array.insert(index_list[i] + i + 1, '-')
        return ''.join(word_array)
