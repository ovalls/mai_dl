import logging
import gensim
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn.manifold import TSNE
import pandas as pd
#import re
import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_input(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            yield gensim.utils.simple_preprocess(line)


# To lemmatize words
wnl = WordNetLemmatizer()

# This function returns the lemma of nouns, verbs, adjectives and adverbs


def lemmatize(pos_word):
    if pos_word[1][0] in {'N', 'V'}:
        return (wnl.lemmatize(pos_word[0], pos=pos_word[1][0].lower()))
    elif pos_word[1][0].startswith('J'):
        return (wnl.lemmatize(pos_word[0], pos='a'))
    elif pos_word[1][0].startswith('R'):
        return (wnl.lemmatize(pos_word[0], pos=pos_word[1][0].lower()))


input_file = 'bjlyrics.txt'
sentences = list(read_input(input_file))    # llista de llistes de paraules que li passaré a word2vec
logging.info("Done reading data file")

print('** Sentences **\n{}'.format(sentences))

sentences_lemmas = []
for s in sentences:
    pairs = nltk.pos_tag(s)
    lemmas = [lemmatize(pair) for pair in pairs]
    lemmas_clean = [l for l in lemmas if l is not None]
    sentences_lemmas.append(lemmas_clean)

print('** Sentences to Lemmas **\n{}'.format(sentences_lemmas))

word_list = [w for words in sentences for w in words]
print('** Set words **\n {}'.format(sorted(set(word_list))))
lemma_list = [l for lemmas in sentences_lemmas for l in lemmas]
print('** Set lemmas **\n {}'.format(sorted(set(lemma_list))))

##### Model for word sentences ###
model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=20, workers=10)
print('Vocabulary for word sentences built')
model.train(sentences, total_examples=len(sentences), epochs=10)
print('Model for word sentences trained')

##### Model for lemma sentences ###
model_lemma = gensim.models.Word2Vec(sentences_lemmas, size=300, window=5, min_count=20, workers=10)
print('Vocabulary for lemma sentences built')
model_lemma.train(sentences_lemmas, total_examples=len(sentences_lemmas), epochs=10)
print('Model for word sentences trained')

# Testing Model for word sentences
print('\nTesting Model for word sentences ')

w = ["man"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["woman"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["guitar"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["song"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["love"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["kiss"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["tonight"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["goodbye"]
simil = model.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))

# Testing Model for lemma sentences
print('\nTesting Model for lemma sentences ')

w = ["man"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["woman"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["guitar"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["song"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["love"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["kiss"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["tonight"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))
w = ["goodbye"]
simil = model_lemma.wv.most_similar(positive=w, topn=5)
print('Most similar to {}: {}'.format(w, simil))

###
# Compress the word vectors into 2D space and plot them
###

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
word_vectors = model.wv
lemma_vectors = model_lemma.wv

# Word labels in gensim's word2vec
# model.wv.vocab is a dict of {word: object of numeric vector}.
# To load the data into X for t-SNE, I made one change.
vocab = list(model.wv.vocab)
X = model[vocab]
vocab_lemma = list(model_lemma.wv.vocab)
X_lemma = model_lemma[vocab_lemma]
# This accomplishes two things: (1) it gets you a standalone vocab list
# for the final dataframe to plot, and (2) when you index model, you can be sure
# that you know the order of the words.
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
X_lemma_tsne = tsne.fit_transform(X_lemma)

# Now let's put X_tsne together with the vocab list.
# This is easy with pandas, so import pandas as pd if you don't have that yet.
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
df_lemma = pd.DataFrame(X_lemma_tsne, index=vocab_lemma, columns=['x', 'y'])
# The vocab words are the indices of the dataframe now.

print('** Head of vocabulary for words **')
print(df.head())
print('** Head of vocabulary for lemmas **')
print(df_lemma.head())

# Scatterplot
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.scatter(df['x'], df['y'], c='blue')
ax2.scatter(df_lemma['x'], df_lemma['y'], c='red')

# Lastly, the annotate method will label coordinates.
# The first two arguments are the text label and the 2-tuple.
# Using iterrows(), this can be very succinct:
for word, pos in df.iterrows():
    ax1.annotate(word, pos)
for lemma, pos in df_lemma.iterrows():
    ax2.annotate(lemma, pos)

plt.show()
