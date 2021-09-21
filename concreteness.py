import os
import pickle

import fasttext
import logging

class WordConcreteness:
	def __init__(self,
				 model = 'svr',
				 model_dir = 'models/',
				 fasttext_model ='cc.nl.300.bin'):
		self.model = self.get_prediction_model(os.path.join(model_dir, model  + '.p'))
		self.embeddings = self.get_fasttext_model(fasttext_model)
		self.seen_words = dict()

	def get_prediction_model(self, path):
		logging.info('Getting prediction model..')
		if not os.path.exists(path):
			raise FileNotFoundError('Model not found in directory.. \
				Make sure "model" and "model_dir" are specified  correctly.') 
		else:
			model = pickle.load(open(path, 'rb'))
			return model

	def get_fasttext_model(self, fasttext_model):
		logging.info('Getting fasttext model..')
		try:
			embeddings =  fasttext.load_model(fasttext_model)
		except ValueError as e:
			answer = input('Fasttext model cannot be found.. Download the model? [y/n]')
			if answer.lower() == 'y':
				fasttext.util.download_model('nl', if_exists='ignore')  
				embeddings =  fasttext.load_model(fasttext_model)
			else:
				raise ValueError()
		return embeddings

	def score(self, word):
		if word not in self.seen_words:
			word_embedding = self.embeddings.get_word_vector(word.lower())
			prediction = round(self.model.predict([word_embedding])[0], 5)
			self.seen_words[word] = prediction
		else:
			prediction = self.seen_words[word]
		return prediction
