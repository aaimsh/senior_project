#!/usr/bin/python3
# -*- coding=utf-8 -*-

import nltk
import gensim
import logging
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.models import load_model
import harakat

import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DocumentProcessor:
    '''
    the main function of this class is to initialize three method:
    the first is to get a word for a giving index
    the second is to get an embedding vector for a giving word
    the third is to get index for a giving embedding vector 
    '''

    def __init__(self,document_path=None, encode='utf-8'):
        '''
        the parameter of init are:
        document_path:- a list contain paths of document, type:--> list(str)
        encode:- the unicode of these documents,          type:--> str
        '''

        self.document_path = document_path
        self.encode = encode
        self.int_to_word = []
        self.word_to_int = {}
        self.word_to_vec = {}
        self.size = 0
        self.corpus = []
        self.classifier = None
        self.nlp_model = None
        self.vec_num = 0

    def process_document(self, vec_num=32, mode=1, count_num=10, token=False):
        '''
        this is the main method, it will train the data to initalize int_to_word, word_to_vec
        and to initialize the ANN model and train it.

        the parameter of process_document are:
        vec_num:- is dimension of embedding vector,          type--> int
        count_num:- the amount of word that should be delet, type--> int

        ***Note***
        if token is false the special token will be ignored
        if token is true two additional token will be added (START,END) to separate the sentences
        '''

        #preparing the documents
        self.vec_num = vec_num
        sentences = []
        sentence = ['START']
        special = ['!','@','#','$','%','^','&','*','(',')','-','+','=','|','/','\\','{','}',
        ';',':','`','~','"',"'",',','.']
        for document in self.document_path:
            try:
                file = open(document, encoding=self.encode)
                line = file.readline()
            except:
                continue
            while(line != ''):
                words = nltk.word_tokenize(line)
                #cleaning the sentences
                if token:
                    for w in words:
                        if w.isdecimal():
                            continue
                        else:
                            if w == '.':
                                sentence.append('END')
                                sentences.append(sentence)
                                sentence = ['START']
                            else:
                                w = harakat.strip_tashkeel(w)
                                sentence.append(w) 
                else:
                    for w in words:
                        if w in special:
                            continue
                        elif w.isdecimal():
                            continue
                        else:
                            sentence.append(w)
                            if len(sentence) == 5:
                                sentences.append(sentence)  
                try:
                    line = file.readline()
                except:
                    continue
            if len(sentence) != 0:
                sentences.append(sentence)
            print(document, 'done')

        self.nlp_model = gensim.models.Word2Vec(sentences, size=vec_num, sg=mode, min_count=count_num)
        for i in sentences:
            for j in i:
                try:
                    self.word_to_vec[j] = self.nlp_model[j]
                    self.corpus.append(j)
                except:
                    continue
        for w in self.word_to_vec:
            self.int_to_word.append(w)
        self.word_to_int = {w:i for i,w in enumerate(self.int_to_word)}
        self.size = len(self.int_to_word)
    
    def get_corpus_as_vec(self, data=None):
        '''
        this method is to get a corpus as an embedded vector for training use
        if data=None the data will be opject corpus

        return --> list(list(float))
        '''

        if data == None:
            data = self.corpus
        corpus = []
        for d in data:
            try:
                w = self.word_to_vec[d]
                corpus.append(w)
            except:
                corpus.append(self.word_to_vec['START'])
        return corpus
    
    def get_corpus_as_index(self, data=None):
        '''
        this method is to get a corpus as index of the words
        if data=None the data will be opject corpus

        return --> list(int)
        '''

        if data == None:
            data = self.corpus
        corpus = []
        for d in data:
            try:
                w = self.word_to_int[d]
                corpus.append(w)
            except:
                corpus.append(0)
        return corpus
    
    def get_corpus_as_oneHotVec(self, data=None):
        '''
        this method is to get a corpus as one-hot-vector for training
        if data=None the data will be opject corpus

        return --> list(list(float))
        '''

        if data == None:
            data = self.corpus
        corpus = []
        for d in data:
            try:
                temp = self.word_to_int[d]
            except:
                temp = 0
            w = one_hot_vec(temp, self.size)
            corpus.append(w)
        return corpus

    def get_training_data(self, t_data=None, time_step=5, mode=1):
        '''
        this metod is for get the training data the input and the target for RNN use
        if data=None the data will be opject corpus,
        if mode = 1:-
            input --> embedded vector
            target --> one-hot-vector
        else:-
            input --> words
            target --> index
        '''
        x = []
        y = []
        if mode == 1:
            if t_data == None:
                x = self.get_corpus_as_vec(data=self.corpus[:-1])
                y = self.get_corpus_as_oneHotVec(data=self.corpus[1:])
            else:
                x = self.get_corpus_as_vec(data=t_data[:-1])
                y = self.get_corpus_as_oneHotVec(data=t_data[1:])

        else:
            if t_data == None:
                x = self.corpus[:-1]
                y = self.get_corpus_as_index(data=self.corpus[1:])
            else:
                x = t_data[:-1]
                y = self.get_corpus_as_index(t_data[1:])
        
        x_train = []
        y_train = []
        for i in range(len(x)-time_step):
            x_train.append(x[i:i+time_step])
            y_train.append(y[i:i+time_step])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return (x_train, y_train)
    
    def convert_words(self, words, time_step=5):
        '''
        this method is to convert a giving words to process data to be use
        in RNN model

        parameters:
        words:- a list of word
        time_step:- the time_step for the RNN backpropagation
        '''
        x = self.get_corpus_as_vec(data=words)
        x_input = []
        for i in range(len(x)-time_step):
            x_input.append(x[i:i+time_step])
        x_input.append(x[len(x)-time_step:])
        x_input = np.array(x_input)
        return x_input

    def save(self, path='', name=''):
        '''
        this method is for saving the attributes and the ANN model
        the attributes are saved in file named my_param

        save has only one parameter which is the path where the files should be saved
        ***note***
        the path must end with "/"
        '''

        saving = (self.document_path, self.encode, self.int_to_word, self.word_to_vec, self.word_to_int, self.size, self.vec_num)
        file = open(path+name,'wb')
        pickle.dump(saving, file)
        file.close()
        
    def load(self, path='', name=''):
        '''
        this method is for loading the attributes and the ANN model
        the attributes are loaded from file named my_param

        load has only one parameter which is the path where the files should are seved
        ***note***
        the path must end with "\"
        '''

        file = open(path+name,'rb')
        self.document_path, self.encode, self.int_to_word, self.word_to_vec, self.word_to_int, self.size, self.vec_num = pickle.load(file)
        file.close()

class WordGenerator:
    '''
    the main function of this class is to initialize the RNN mode:
    for natural language and train it to generate words
    '''

    def __init__(self, x_shape=(1,1), y_shape=1, rnn_unit=100, normal_unit=100, normal_layer_n=1):
        '''
        the parameter of init are:
        x_shape:- the shape of the input training data, type:--> tupe(int,int)
        y_shape:- the shape of the target training data, type:--> int
        rnn_unit:- the number of neurons in rnn layer, type:--> int
        normal_unit:- the number of neurons in Dense layer, type:--> int
        normal_layer_n: the number of Dense layer, type:--> int
        '''

        #initialize the RNN model(generator)
        self.generator = Sequential()
        self.generator.add(GRU(units = rnn_unit, input_shape = x_shape, return_sequences=True, name='LSTM'))
        self.generator.add(Dropout(0.1))

        for i in range(normal_layer_n):
            if i == normal_layer_n - 1:
                self.generator.add(Dense(units = y_shape, activation = 'softmax', name='softmax'))
            else:
                self.generator.add(Dense(units = normal_unit, activation = 'tanh', kernel_initializer='glorot_uniform',
                name = 'Dense_{}'.format(i)))
                self.generator.add(Dropout(0.1))
        
        self.generator.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    def train(self, x_train, y_train, batch_n=100, epochs_n=100):
        '''
        this method to train the model

        parameters:
        x_train:- the input training data
        y_train:- the target training data
        batch_n:- the number of batch
        epochs_n:- the number of epochs
        '''
        self.generator.fit(x_train, y_train, epochs=epochs_n, batch_size=batch_n)
    
    def generate(self, data):
        '''
        this method is to generate a word for a giving data
        '''
        return self.generator.predict(data)

    def save(self, path='', name=''):
        '''
        this method is for saving the model under the name gene generator.h5
        ***note***
        the path must end with "\"
        '''
        self.generator.save(path+name+'.h5')

    def load(self, path='', name=''):
        '''
        this method is for load the model under the name gene generator.h5
        ***note***
        the path must end with "\"
        '''
        self.generator = load_model(path+name+'.h5')

class Writer:
    '''
    the main function of this class is the write method
    which generate the new words giving a sequence of words
    '''

    def __init__(self, document=None, generator=None, time_step=None, multi_generator = False):
        '''
        parameters:
        document:- is a DocumentProcessor opject
        generator:- is a WordGenerator opject
        time_step: is the time step for the backpropagation

        ***note***
        if the multi_generator is True please assigne the time_step to the highest time_step, and the
        generator parameter must be as a list of generator opjects
        '''
        
        self.document = document
        self.generator = generator
        self.time_step = time_step
        self.multi_generator = multi_generator

    def write(self, initial_mode=0, words=None, number_of_word=1):
        '''
        the main method which which generate the new words 
        giving a sequence of words.

        parameter:
        initial_mode:- is the mode for the data
        words:- the initial sequences of words
        number_of_word:- is the number of word to generate

        ***not***
        if the mode is 0 no need for a words
        '''
        
        sentences = []
        index = 0
        if initial_mode == 1:
            sentences = words
        if self.multi_generator:
            initial_n_sentence = len(sentences)
            i = initial_n_sentence
            input_word = []
            while i < initial_n_sentence+number_of_word:
                if i == 0:
                    sentences.append(self.random_word())
                    i += 1
                elif i < self.time_step:
                    input_word = self.document.convert_words(sentences[:i], time_step=i)
                    result = self.generator[i-1].generate(input_word)
                    r = result[0][-1]
                    r = np.argmax(r)
                    sentences.append(self.document.int_to_word[r])
                    i += 1
                else:
                    input_word = self.document.convert_words(sentences[i-self.time_step: i], time_step=self.time_step)
                    result = self.generator[-1].generate(input_word)
                    r = result[0][-1]
                    r = np.argmax(r)
                    sentences.append(self.document.int_to_word[r])
                    i += 1
        else:
            for i in range(number_of_word):
                input_word = []
                if len(sentences) < self.time_step:
                    input_word = self.padding(sentences)
                else:
                    input_word = self.document.convert_words(sentences[index:self.time_step+index])
                    index += 1
                result = self.generator.generate(input_word)
                r = result[0][-1]
                r = np.argmax(r)
                sentences.append(self.document.int_to_word[r])
        text = self.get_text(sentences)
        return text

    def get_text(self, word):
        '''
        this method take a list of string and return a string
        '''

        result = ''
        for w in word:
            if w == 'END':
                result += '.'
            elif w == 'START':
                result += ' '
            else:
                result += ' ' + w
        return result

    def random_word(self):
        '''
        this method is to produce a random word from the document

        Return -> str
        '''
        r = random.randrange(self.document.size)
        return self.document.int_to_word[r]

    def padding(self, words):
        '''
        padding the words if the number of words less than time step
        '''

        size = self.time_step - len(words) 
        vector = []
        for i in range(size):
            vector.append('START')
        for w in words:
            vector.append(w)
        return self.document.convert_words(vector)

    def save(self, path=''):
        '''
        this method is for saving opject writer
        '''

        #saving information 
        saving = (self.time_step, self.multi_generator)
        file = open(path+'writer_info','wb')
        pickle.dump(saving, file)
        file.close()
        #saving the document
        self.document.save(path, 'writer_doc')
        #saving the generator
        if self.multi_generator:
            for i in range(self.time_step):
                self.generator[i].save(path,'writer_gen{0}'.format(i+1))
        else:
            self.generator.save(path, 'writer_gen')
    
    def load(self, path=''):
        '''
        this method is for load opject writer
        '''

        #load information 
        file = open(path+'writer_info','rb')
        self.time_step, self.multi_generator = pickle.load(file)
        file.close()
        #load the document
        self.document = DocumentProcessor()
        self.document.load(path, 'writer_doc')
        #load the generator
        if self.multi_generator:
            self.generator = []
            for i in range(self.time_step):
                self.generator.append(WordGenerator())
                self.generator[i].load(path,'writer_gen{0}'.format(i+1))
        else:
            self.generator = WordGenerator()
            self.generator.load(path, 'writer_gen')

def one_hot_vec(index, size):
    '''
    a function to do one-hot encoding

    parameters are:
    word:- list of word,         type--> list(str)
    index:- the index of a word, type--> int

    return a vector, type--> list(int)
    '''
    vec = np.zeros((size,))
    vec[index] = 1
    return vec
