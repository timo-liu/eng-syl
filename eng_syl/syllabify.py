from tensorflow.keras.layers import  Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation, GRU, ZeroPadding1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os


class Syllabel:
    def __init__(self,e2i_ortho ='e2i.pkl', ortho_input_size=45, latent_dim=256, embed_dim=256, max_feat=259):
        self.this_dir, this_filename = os.path.split(__file__)
        path_clean = os.path.join(self.this_dir, 'clean.pkl')
        self.clean = pickle.load(open(path_clean,'rb'))
        path_e2i = os.path.join(self.this_dir, e2i_ortho)
        self.e2i_ortho = pickle.load(open(path_e2i, 'rb'))
        self.ortho_input_size = ortho_input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat = max_feat
        self.model = self.build_model()
        self.model.load_weights(os.path.join(self.this_dir,'syllabler_best_weights.h5'))


    def build_model(self):
        
        # orthographic and ipa input layers
        ortho_inputs = Input(self.ortho_input_size,)
        # first branch ortho
        x = Embedding(self.max_feat, self.embed_dim, input_length=self.ortho_input_size)(ortho_inputs)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer=regularizers.l2(1e-5)), input_shape=(self.ortho_input_size, 1))(x)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout = 0.2, activity_regularizer= regularizers.l2(1e-5) ), input_shape=(self.ortho_input_size, 1))(x)
        # x = TimeDistributed(Dense(3, activation = 'softmax'))(x)
        x = ZeroPadding1D(padding=(0, self.ortho_input_size - x.shape[1]))(x)
        z = TimeDistributed(Dense(3))(x)
        z = Activation('softmax')(z)
        
        model = Model(inputs=[ortho_inputs], outputs=z)
        
        return model
    
    def syllabify(self, word):
        if word in self.clean:
            return self.clean[word]
        inted_ortho = []
        for c in word.lower():
            inted_ortho += [self.e2i_ortho[c]]
            
        
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size, 1), verbose = 0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(word, indexes)
        return converted

    def machine_syllabify(self, word):
        inted_ortho = []
        for c in word.lower():
            inted_ortho += [self.e2i_ortho[c]]


        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size, 1), verbose = 0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(word, indexes)
        return converted

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