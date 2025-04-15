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
        ortho_inputs = Input((self.ortho_input_size,))
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
    
    def syllabify(self, word, return_list = False, save_clean = True):
        if word in self.clean:
            if return_list:
                if isinstance(self.clean[word], list):
                    return self.clean[word]
                else:
                    return self.clean[word].split('-')
            
        else:
            outcome = self.machine_syllabify(word, return_list)
            if save_clean:
                self.clean[word] = outcome
            return outcome

    def machine_syllabify(self, word, return_list = False):
        inted_ortho = []
        non_alpha = []
        non_alpha_indexes = []
        alpha_word = ""
        for i, c in enumerate(word):
            if not c.isalpha():
                non_alpha.append(c)
                non_alpha_indexes.append(i)
            else:
                inted_ortho += [self.e2i_ortho[c.lower()]]
                alpha_word += c


        inted_ortho = pad_sequences([inted_ortho], maxlen=self.ortho_input_size, padding='post')[0]
        predicted = self.model.predict(inted_ortho.reshape(1, self.ortho_input_size, 1), verbose = 0)[0]
        indexes = self.to_ind(predicted)
        converted = self.insert_syl(alpha_word, indexes)
        with_non_alpha = self.reinsert_nonalpha(converted, non_alpha, non_alpha_indexes, indexes)
        if not return_list:
            return with_non_alpha
        else:
            return self.split_by_syllable(with_non_alpha, non_alpha_indexes, indexes)

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

    def reinsert_nonalpha(self, word, non_alpha, non_alpha_indexes, syl_indexes): # word is already hyphenated
        if non_alpha:
            inserted = 0
            word_array = [*word]
            
            for c,i in zip(non_alpha, non_alpha_indexes):
                insert_nums = len(np.where(np.array(syl_indexes[:i-inserted]) == 2)[0])
                word_array.insert(i + insert_nums, c)
                inserted += 1
            return ''.join(word_array)
        else:
            return word

    def split_by_syllable(self, word, non_alpha_indexes, syl_indexes): # word is the reinserted with non_alphas string
            if not non_alpha_indexes:
                return word.split('-')
            else:
                orig_index_list = np.where(np.array(syl_indexes) == 2)[0]
                new_index_list = []
                for index in orig_index_list:
                    filtered = [a for i, a in enumerate(non_alpha_indexes) if a - i <= index]
                    new_index_list.append(index + len(filtered))
                
                # Split the word based on the adjusted indices
                prev_index = 0
                result = []
                
                for j, index in enumerate(new_index_list, start = 1):
                    result.append(word[prev_index:index + j])
                    prev_index = index + j
                
                # Add the remaining part of the string
                result.append(word[prev_index:])
                
                return result

    def evaluate_english_validity(self, syllable):
        '''
        Evaluate whether or not the syllable is likely to be English pronounceable
        Returns: onced up words or False
        '''
        onsets = ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sk', 'sl', 'sm', 'sn',
     'sp', 'st', 'str', 'sw', 'tr', 'ch', 'sh', 'm', 'c', 'b', 'r', 'd', 'h', 's', 'p', 'l', 'g', 'f', 'w', 't', 'k', 'n', 'v', 'st', 'pr', 'j', 'br', 'ch', 'gr', 'sh',
  'tr', 'cr', 'fr', 'z', 'sp', 'wh', 'cl', 'y', 'bl', 'th', 'fl', 'sch', 'pl', 'q', 'dr', 'str', 'sc', 'sl', 'kr', 'sw', 'gl',
  'ph', 'kl', 'sm', 'sn', 'kn', 'sk', 'mcc', 'scr', 'wr', 'mc', 'chr', 'spr', 'thr', 'tw', 'schw', 'mcg', 'mck', 'rh',
  'sq', 'schl', 'shr', 'schr', 'x', 'schm', 'mcm', 'gh', 'mcn', 'hyp', 'mccl', 'schn', 'mcd', 'hydr', 'kh', 'ts',
  'mcl', 'spl', 'dw', 'pf', 'mccr', 'mcf', 'typ', 'cz', 'sr', 'cycl', 'gn', 'hr', 'hy', 'syn', 'sz', 'kw', 'dyn', 'phys', 'symb', 'dyn', 'symb']

        nuclei = ['a', 'e', 'i', 'o', 'u', 'oo', 'ia', 'ie', 'ee', 'io', 'au', 'ea', 'ou', 'ai', 'ue', 'ei', 'eau', 'eu', 'oe', 'ae', 'eo',
  'oa', 'oo', 'ao', 'ua', 'oi', 'ui', 'aa', 'ieu', 'uo', 'oia', 'aue', 'iu', 'aia', 'iou', 'ii', 'aio', 'uie', 'eia', 'iao' ,'y', 'uh', 
          'ay', 'ey','ah','eh','oh','oy', 'aigh', 'igh', 'eigh', 'aw', 'ow', 'ew', 'ye', 'ooh', 'owe', 'awe', 'ore', 'er', 'or', 'ere', 
          'are', 'ar', 'ur', 'ir', 'ire', 'ue', 'eye', 'aye', 'ye', 'uy']

        onsets = sorted(sorted(set(onsets)),key=len, reverse=True)
        nuclei = sorted(sorted(set(nuclei)),key=len, reverse=True)

        for i in onsets:
            if syllable.startswith(i):
                i_less = syllable.replace(i, '', 1)
                for n in nuclei:
                    if i_less.startswith(n):
                        n_less = i_less.replace(n, '')
                        return i + '-' + n + '-' + n_less
            break
        for n in nuclei:
                if syllable.startswith(n):
                    i = ''
                    n_less = syllable.replace(n, '')
                    return  n + '-' + n_less
                elif n in syllable:
                    onset, coda = syllable.split(n, 1)
                    return '-'.join([onset, n, coda])
        return False

