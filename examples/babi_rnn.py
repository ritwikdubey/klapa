from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

def tokenize(sent):
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
	data = []
	story = []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ',1)
		nid = int(nid)
		if nid == 1:
			story = []
		if '\t' in line:
			q, a, supporting = line.split('\t')
			q = tokenize(q)
			if only_supporting:
				supporting = map(int, supporting.split())
				substory = [story[i - 1] for i in supporting]
			else:
				substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenize(line)
			sent.append(sent)
	return data

def get_stories(f, only_supporting=False, max_length=None):
	data = parse_stories(f.readlines(), only_supporting