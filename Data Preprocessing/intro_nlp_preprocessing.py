from nltk.corpus import inaugural, brown,stopwords
from urllib import request
from nltk import ConditionalFreqDist
	
def calculateWordCounts(text):
    print(len(text))
    print(len(set(text)))
    print(int(len(text)/(len(set(text)))))
	
def filterWords(text):
    ing_words = [word for word in set(text) if word.endswith("ing")]
    large_words = [word for word in text if len(word) > 15 ]
    upper_words = [word for word in set(text) if word.isupper()]
    
    return ing_words, large_words, upper_words
	
def findWordFreq(text, word):
    textfreq = nltk.FreqDist([word for word in text if word.isalpha()])
    wordfreq = textfreq[word]
    maxfreq = textfreq.most_common(1)[0][0]
    
    return wordfreq, maxfreq

def accessTextCorpora(fileid, word):
    n_words = len(inaugural.words(fileid))
    n_unique_words = len(set(inaugural.words(fileid)))
    wordcoverage = int(n_words/n_unique_words)
    ed_words = [word for word in set(inaugural.words(fileid)) if word.endswith("ed")]  
    lower_words = [word.lower() for word in inaugural.words(fileid) if word.isalpha()]
    textfreq = nltk.FreqDist(lower_words)
    wordfreq = textfreq[word]
    
    return wordcoverage, ed_words, wordfreq
	
def createUserTextCorpora(filecontent1, filecontent2):
    with open(r'nltk_data/content1.txt', 'w') as f:
        f.write(filecontent1)
        
    with open(r'nltk_data/content2.txt', 'w') as f:
        f.write(filecontent2)
        
    from nltk.corpus import PlaintextCorpusReader
    
    text_corpus = PlaintextCorpusReader('nltk_data', '.*')
    
    no_of_words_corpus1 = len(text_corpus.words('content1.txt'))
    no_of_unique_words_corpus1 = len(set(text_corpus.words('content1.txt')))
    
    no_of_words_corpus2 = len(text_corpus.words('content2.txt'))
    no_of_unique_words_corpus2 = len(set(text_corpus.words('content2.txt')))
    
    return text_corpus, no_of_words_corpus1, no_of_unique_words_corpus1, no_of_words_corpus2, no_of_unique_words_corpus2
	

def calculateCFD(cfdconditions, cfdevents):
    stopword = set(stopwords.words('english'))
    cdev_cfd = nltk.ConditionalFreqDist([(genre, word.lower()) for genre in cfdconditions for word in brown.words(categories=genre) if not word.lower()  in stopword])
    cdev_cfd.tabulate(conditions = cfdconditions, samples = cfdevents)
    inged_cfd = [ (genre, word.lower()) for genre in brown.categories() for word in brown.words(categories=genre) if (word.lower().endswith('ing') or word.lower().endswith('ed')) ]
    inged_cfd = [list(x) for x in inged_cfd]
    
    for wd in inged_cfd:
        if wd[1].endswith('ing') and wd[1] not in stopword:
            wd[1] = 'ing'
        elif wd[1].endswith('ed') and wd[1] not in stopword:
            wd[1] = 'ed'
    #print(inged_cfd)
    inged_cfd = nltk.ConditionalFreqDist(inged_cfd)
    #print(inged_cfd.conditions())    
    inged_cfd.tabulate(conditions=cfdconditions, samples = ['ed','ing'])
	

def processRawText(textURL):
    textcontent = request.urlopen(textURL).read()
    decoded = textcontent.decode('unicode_escape') 
    tokenizedlcwords = [word.lower() for word in nltk.word_tokenize(decoded)]
    noofwords = len(tokenizedlcwords)
    noofunqwords = len(set(tokenizedlcwords))
    wordcov = int(noofwords/noofunqwords)
    
    wordfreq = nltk.FreqDist([x for x in tokenizedlcwords if x.isalpha()])
    maxfreq = wordfreq.most_common(1)[0][0]
    
    return noofwords, noofunqwords, wordcov, maxfreq
	
def performBigramsAndCollocations(textcontent, word):

    tokenizedword = nltk.regexp_tokenize(textcontent, pattern = r'\w*', gaps = False)
    tokenizedwords = [x.lower() for x in tokenizedword if x != '']
    tokenizedwordsbigrams=nltk.bigrams(tokenizedwords)
    stop_words= stopwords.words('english')
    tokenizednonstopwordsbigrams=[(w1,w2) for w1 , w2 in tokenizedwordsbigrams if (w1 not in stop_words and w2 not in stop_words)]
    cfd_bigrams=nltk.ConditionalFreqDist(tokenizednonstopwordsbigrams)
    mostfrequentwordafter=cfd_bigrams[word].most_common(3)
    tokenizedwords = nltk.Text(tokenizedwords)
    collocationwords = tokenizedwords.collocation_list()

    return mostfrequentwordafter ,collocationwords
	
def performStemAndLemma(textcontent):
    tokenizedword = nltk.regexp_tokenize(textcontent, pattern = r'\w*', gaps = False)
    tokenizedwords = [y for y in tokenizedword if y != '']
    unique_tokenizedwords = set(tokenizedwords)
    tokenizedwords = [x.lower() for x in unique_tokenizedwords if x != '']
    stop_words = set(stopwords.words('english')) 
    filteredwords = []
    for x in tokenizedwords:
        if x not in stop_words:
            filteredwords.append(x)
    ps = nltk.stem.PorterStemmer()
    ls = nltk.stem.LancasterStemmer()
    wnl = nltk.stem.WordNetLemmatizer()
    porterstemmedwords =[]
    lancasterstemmedwords = []
    lemmatizedwords = []
    for x in filteredwords:
        porterstemmedwords.append(ps.stem(x))
        lancasterstemmedwords.append(ls.stem(x))
        lemmatizedwords.append(wnl.lemmatize(x))
    return porterstemmedwords, lancasterstemmedwords, lemmatizedwords


def tagPOS(textcontent, taggedtextcontent, defined_tags):
    # Write your code here
    words = nltk.word_tokenize(textcontent)
    nltk_pos_tags = nltk.pos_tag(words)
    
    tagged_pos_tag = [ nltk.tag.str2tuple(word) for word in taggedtextcontent.split()]
    tagger = nltk.UnigramTagger(model=defined_tags)
    
    unigram_pos_tag = tagger.tag(words)
    
    return nltk_pos_tags, tagged_pos_tag, unigram_pos_tag

