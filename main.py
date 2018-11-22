import os
import subprocess 
import sys
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
import ast
import nltk
import pickle
from nltk.corpus import stopwords
from nltk import *

concepts=[]
stemmed_concepts = {}
concept_map = set()
model = {}
connections = set()

stemmer = SnowballStemmer("english")

#string list to list
def getList(listString):
	x = ast.literal_eval(listString)
	return x


# 1st Tokenize and form sentence
def tokenizeSentence(fileContent):
	"""
	:param fileContent: fileContent is the file content which needs to be summarized
	:return: Returns a list of string(sentences)
	"""
	return nltk.sent_tokenize(fileContent.decode('utf-8'))


# 2nd Case Folding
def caseFolding(line):
	"""
	:param line is the input on which case folding needs to be done
	:return: Line with all characters in lower case
	"""
	return line.lower()


# 3rd Tokenize and form tokens from sentence
def tokenizeLine(sentence):
	"""
	:param sentence: Sentence is the english sentence of the file
	:return: List of tokens
	"""
	return nltk.word_tokenize(sentence)


def cosineSimilarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0.0, 0.0, 0.0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return (float)(sumxy)/math.sqrt(sumxx*sumyy)



def loadGloveModel(gloveFile):
	"""
	:param : glove file contains vectors representation of almost all words.
	:Add the representaion of words in dict i.e. model
	"""
	model = {} 
	f = open(gloveFile,'r')
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	with open('model.pickle', 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return model



def build_concept_map(subjects, _objects, predicate, mapSim):
	"""
    :params: tokenized words in subjects and objects. And predicate used b/w subject and object in sentence
    :Add corresponding relation b/w concepts found in subjects and objects
    """
	for sub in subjects:
		if stemmer.stem(sub) in stemmed_concepts:
			sub_val = stemmed_concepts[stemmer.stem(sub)]
			for obj in _objects:
				if stemmer.stem(obj) in stemmed_concepts:
					obj_val = stemmed_concepts[stemmer.stem(obj)]
					if (sub, obj) in mapSim and mapSim[(sub, obj)] >= 0.05 and (sub, obj) not in connections:
						connections.add((sub, obj))
						concept_map.add((sub, obj, predicate, mapSim[(sub, obj)]))
					if (obj, sub) in mapSim and mapSim[(obj, sub)] >= 0.05 and (sub, obj) not in connections:
						connections.add((sub, obj))
						concept_map.add((sub, obj, predicate, mapSim[(obj, sub)]))



# for visualizing concept map
def make_graph(concept_map):
	""" 
	making use o graphviz
	"""
	output_dot_file = open("graph.dot", "w")
	output_dot_file.write('digraph G {\n\n')
 
	concepts = set()
	for con in concept_map:
		concepts.add(con[0])
		concepts.add(con[1])

	# for drawing each vertex in the graph
	for con in concepts:
		output_dot_file.write('  %s [shape = box]\n' % str(con))


    # for drawing each edges
	output_dot_file.write('\n')
	for con in concept_map:
		output_dot_file.write('  %s -> %s [color=firebrick,penwidth=%s, label="%s"]\n' % (con[0], con[1], str(con[3]*3.0 + 1.0), str(con[2])))

    # rendering .dot file to generate .png file
	output_dot_file.write('\n}\n')
	output_dot_file.close()
    
	args = ['circo', '-Goverlap=scale', 'graph.dot', '-Tpng', '-o', 'output_image.png']
	subprocess.Popen(args,stdin=subprocess.PIPE, stdout=subprocess.PIPE)



def main():
	global model
	global concepts
	global concept_map
	global stemmed_concept

	# loading pre-traing word2vec
	if os.path.isfile('model.pickle'):
		handle = open('model.pickle', 'rb')
		model = pickle.load(handle)
	else:
		model = loadGloveModel('./glove/glove.6B.300d.txt')

	content = None
	openIEList = None
	if len(sys.argv)>1:
		input_file = sys.argv[1]
	else:
		input_file = 'input.txt'
	with open(input_file, 'r') as content_file:
		content = content_file.read()

	os.system('curl -s http://api.dbpedia-spotlight.org/en/annotate --data-urlencode \"text='+content+'" --data \"confidence=0.35&support=10\" -H \"Accept: application/json\"  > data.json')

	data = json.load(open('data.json'))

	for i in range (0,len(data['Resources'])):
		concept_word = data['Resources'][i]['@surfaceForm'].lower().encode('ascii','ignore')
		concepts.append(concept_word)
		concept_word_list = nltk.word_tokenize(concept_word)
		for word in concept_word_list:
			stemmed_concepts[stemmer.stem(word)] = concept_word

	toBeDeleted=[]
	mapSim = {}
	 

	for i,concept1 in enumerate(concepts):
		words_of_concept1 = nltk.word_tokenize(concept1)
		A = []
		a_list = []
		for word in words_of_concept1:
			if word in model:
				a_list.append(model[word].tolist())
		if len(a_list) >=1:
			for k in range(len(a_list[0])):
				c1 = 0.0
				for a in a_list:
					c1 += a[k]
				A.append(c1/(float)(len(a_list)))

			for j,concept2 in enumerate(concepts):
				if j>i:
					key = (concept1, concept2)
					words_of_concept2 = nltk.word_tokenize(concept2)		
					B = []
					b_list = []
					for word in words_of_concept2:
						if word in model:
							b_list.append(model[word].tolist())
					if len(b_list)>=1:
						for k in range(len(b_list[0])):
							c2 = 0.0
							for b in b_list:
								c2 += b[k]
							B.append(c2/(float)(len(b_list)))
						mapSim[key] = cosineSimilarity(A, B)
	

	os.system('cp '+ input_file + ' ./OpenIE/')
	os.system('python ./OpenIE/main.py -f ./' + input_file + ' > output.json')

	openIEList=None

	with open('output.json', 'r') as content_file:
		openIEList = content_file.read()


	triplets = getList(openIEList.split('\n')[1])

	for triplet in triplets:
		subject, predicate, _object = triplet[0], triplet[1], triplet[2]

		subjects = tokenizeLine(subject.lower())
		objects  = tokenizeLine(_object.lower())

		build_concept_map(subjects, objects, predicate, mapSim)
	
	print "*******************"
	make_graph(concept_map)
	

if __name__ == '__main__':
	main()


