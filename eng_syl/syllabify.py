from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from eng_syl.BahdanauAttention import AttentionLayer
import numpy as np
import pickle

class wordSegmenter:
    def __init__(self, max_encoder_len=34, max_decoder_len=44, latent_dim = 500, embedding_dim=500):

        e2i_file = open('e2i_w2s.pkl', 'rb')
        d2i_file = open('d2i_w2s.pkl', 'rb')

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.e2i = pickle.load(e2i_file)
        self.i2e = {v: k for k, v in self.e2i.items()}
        self.d2i = pickle.load(d2i_file)
        self.i2d = {v: k for k, v in self.d2i.items()}
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.num_encoder_vocab = len(self.e2i) + 1
        self.num_decoder_vocab = len(self.d2i) + 1
    
    def build_training_model(self):
        self.encoder_inputs = Input(shape=(self.max_encoder_len,))
        self.enc_emb = Embedding(self.num_encoder_vocab, self.embedding_dim, trainable=True)(self.encoder_inputs)

        # Bidirectional lstm layer
        self.enc_lstm1 = Bidirectional(LSTM(self.latent_dim, return_sequences=True, return_state=True))
        self.encoder_outputs1, self.forw_state_h, self.forw_state_c, self.back_state_h, self.back_state_c = self.enc_lstm1(
            self.enc_emb)

        # Concatenate both h and c
        self.final_enc_h = Concatenate()([self.forw_state_h, self.back_state_h])
        self.final_enc_c = Concatenate()([self.forw_state_c, self.back_state_c])

        # get Context vector
        self.encoder_states = [self.final_enc_h, self.final_enc_c]
        
        self.decoder_inputs = Input(shape=(None,))

        # decoder embedding with same number as encoder embedding
        self.dec_emb_layer = Embedding(self.num_decoder_vocab, self.embedding_dim)
        self.dec_emb = self.dec_emb_layer(
            self.decoder_inputs)  # apply this way because we need embedding layer for prediction

        # In encoder we used Bidirectional so it's having two LSTM's so we have to take double units(256*2=512) for single decoder lstm
        # LSTM using encoder's final states as initial state
        self.decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.dec_emb, initial_state=self.encoder_states)

        # Using Attention Layer
        self.attention_layer = AttentionLayer()
        self.attention_result, self.attention_weights = self.attention_layer(
            [self.encoder_outputs1, self.decoder_outputs])

        # Concat attention output and decoder LSTM output
        self.decoder_concat_input = Concatenate(axis=-1, name='concat_layer')(
            [self.decoder_outputs, self.attention_result])

        # Dense layer with softmax
        self.decoder_dense = Dense(self.num_decoder_vocab, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_concat_input)
        
        self.training_model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
    def compile(self):
        self.training_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    def fit(self, x_tr, y_tr_in, y_tr_out, x_test, y_test_in, y_test_out, ep, batch_size, save_filename):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        ck = ModelCheckpoint(filepath=save_filename, monitor='val_acc', verbose=1, save_best_only=True,
                             mode='max')
        Callbacks = [es, ck]
        self.training_model.fit([x_tr, y_tr_in], y_tr_out, epochs=ep, callbacks=Callbacks, batch_size=batch_size,
                                validation_data=(([x_test, y_test_in]), y_test_out))

    def load_model_weights(self, filename = 'Segmenter2.0_bestweights.h5'):
        self.training_model.load_weights(filename)

    def build_inference_model(self):
        self.encoder_model_inference = Model(self.encoder_inputs,
                                             outputs=[self.encoder_outputs1, self.final_enc_h, self.final_enc_c])

        # Decoder Inference
        self.decoder_state_h = Input(
            shape=(self.latent_dim * 2,))  # This numbers has to be same as units of lstm's on which model is trained
        self.decoder_state_c = Input(shape=(self.latent_dim * 2,))

        # we need hidden state for attention layer
        # 36 is maximum length if english sentence It has to same as input taken by attention layer can see in model plot
        self.decoder_hidden_state_input = Input(shape=(self.max_encoder_len, self.latent_dim * 2))
        # get decoder states
        self.dec_states = [self.decoder_state_h, self.decoder_state_c]

        # embedding layer
        self.dec_emb2 = self.dec_emb_layer(self.decoder_inputs)
        self.decoder_outputs2, self.state_h2, self.state_c2 = self.decoder_lstm(self.dec_emb2,
                                                                                initial_state=self.dec_states)

        # Attention inference
        self.attention_result_inf, self.attention_weights_inf = self.attention_layer(
            [self.decoder_hidden_state_input, self.decoder_outputs2])
        self.decoder_concat_input_inf = Concatenate(axis=-1, name='concat_layer')(
            [self.decoder_outputs2, self.attention_result_inf])

        self.dec_states2 = [self.state_h2, self.state_c2]
        self.decoder_outputs2 = self.decoder_dense(self.decoder_concat_input_inf)

        # get decoder model
        self.decoder_model_inference = Model(
            [self.decoder_inputs] + [self.decoder_hidden_state_input, self.decoder_state_h, self.decoder_state_c],
            [self.decoder_outputs2] + self.dec_states2)
    
    def decode_sequence(self, input_seq):
        e_out, e_h, e_c = self.encoder_model_inference.predict(input_seq, verbose=0)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.d2i['<']

        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            (output_tokens, h, c) = self.decoder_model_inference.predict([target_seq] + [e_out, e_h, e_c], verbose=0)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.i2d[sampled_token_index]

            if sampled_token != '>':
                decoded_sentence += [sampled_token]

            # Exit condition: either hit max length or find the stop word.
            if (sampled_token == '>') or (len(decoded_sentence) >= self.max_decoder_len):
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            (e_h, e_c) = (h, c)
        return decoded_sentence

    def word2seq(self, input_word):
        final_seq = []
        for c in input_word:
            final_seq += [self.e2i[c]]
        final_seq = pad_sequences([final_seq], maxlen=self.max_encoder_len, padding='post')[0]
        return final_seq

    def translate(self, input_word):
        seq = self.word2seq(input_word).reshape(1, self.max_encoder_len)
        return self.decode_sequence(seq)