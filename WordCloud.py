import os
import nltk
from nltk.probability import FreqDist
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')

def create_wordcloud(text, stopwords, filename, apply_mask=None):
    if apply_mask is not None:
        wordcloud = WordCloud(
            background_color="white",
            width=1000,
            height=500,
            collocations=False,
            max_words=2000,
            mask=apply_mask,
            stopwords=stopwords,
            min_font_size=10,
            max_font_size=100)
        wordcloud.generate(text)
        wordcloud.to_file(filename)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.figure()
        plt.imshow(apply_mask, cmap=plt.cm.gray, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    else:
        wordcloud = WordCloud(
            min_font_size=10,
            max_font_size=100,
            stopwords=stopwords,
            width=1000,
            height=1000,
            max_words=1000,
            background_color="white").generate(text)

        wordcloud.to_file(filename)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

def compile_stopwords_list_frequency(words):
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    stopwords = [tup[0] for tup in sorted_words if tup[1] > 100]
    return stopwords

# Assuming loadData is a custom function to read file contents
def loadData(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

text = loadData(r'C:\Users\Cedric Sulisthio\OneDrive\Documents\IBEProjectsandTutor\Artificial Intelligence for Business\mytext.txt')
text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)

# Compile custom stopwords based on frequency
custom_stopwords = compile_stopwords_list_frequency(words)

# Filter words based on custom stopwords
words = [word for word in words if word.lower() not in custom_stopwords]
filtered_text = ' '.join(words)

# Create word cloud
create_wordcloud(filtered_text, custom_stopwords, "myWordCloud.png")