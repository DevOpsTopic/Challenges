
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim import models
from gensim.models import LdaModel, CoherenceModel
from gensim.models.wrappers import LdaMallet
import pprint

# spacy for lemmatization
import spacy
from spacy.lang.en import English
import pyLDAvis
#import pyLDAvis.gensim_models

# Plotting tools
import pyLDAvis
#import pyLDAvis.gensim 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv("F:\\Project\\DevOps_Dataset.csv",encoding="latin-1")
reviews_datasets.dropna()
obody = reviews_datasets['PostTitle']

reviews_datasets.head()


from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk

import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer

tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

# Preprocess the text for vectorization
# - Remove HTML
# - Remove stopwords
# - Remove special characters
# - Convert to lowercase
# - Stemming




import re

# Convert to list
data = reviews_datasets.PostTitle.values.tolist()

# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]

# Remove distracting single and double quotes
data = [re.sub("\'", "", sent) for sent in data]
data = [re.sub('\", "', '', sent) for sent in data]
data = [re.sub('\\"', '', sent) for sent in data]
data = [re.sub('\"', '', sent) for sent in data]
data = [re.sub('[\\:"]', '', sent) for sent in data]

# Remove web links
data = [re.sub(r'^https?:\/\/.*[\r\n]*', '', sent) for sent in data]

print(data[:3])


# In[32]:


# Tokenize words and text clean up
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:3])



# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=75) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=75)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[3]]])


# In[34]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out




data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#spacy.load('en_core_web_sm')

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:2])

#reviews_datasets['topics']=data_lemmatized
#reviews_datasets['obody']=obody
#reviews_datasets.head(20)
#reset option to default value




# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
#mallet_path = 'D:/mallet-2.0.8/bin/mallet' 
# update this path
#lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=15, id2word=id2word, optimize_interval=10, iterations=1000)



import os
from gensim.models.wrappers import LdaMallet

os.environ['MALLET_HOME'] = 'D:\\mallet\\mallet-2.0.8'

mallet_path = 'D:\\mallet\\mallet-2.0.8\\bin\\mallet'

iteration_item=[100,500,1000,1500]
num_topic=[5,10,15,20,25,30,35,40,45,50]
alpha_value=[.05,.1,.5,1,5,10,50]

model_results = {'Iteration': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

for iteration in iteration_item:
    for topic in num_topic:
        for alpha in alpha_value:
            print(topic)
            lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=topic, id2word=id2word, optimize_interval=10, alpha=alpha, iterations=iteration)
            # Compute Coherence Score
            coherence_model_ldamallet = CoherenceModel(model=lda_mallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_ldamallet = coherence_model_ldamallet.get_coherence()
            
                       model_results['Iteration'].append(iteration)
            model_results['Topics'].append(topic)
            model_results['Alpha'].append(alpha)
            model_results['Beta'].append('0.01')
            model_results['Coherence'].append(coherence_ldamallet)
            print ("Topic:",topic, "Alha:", alpha, "Coherence:",coherence_ldamallet)
                
          
        
pd.DataFrame(model_results).to_csv('F:\\Project\\coherencevalue_score.csv', index=False)





