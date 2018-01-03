#!/usr/bin/env python3
import sys
sys.path.append('/datadrive/nlp/jasper/w2v/godin/word2vec_twitter_model/')
from word2vecReader import Word2Vec
from nltk.corpus import words
import io
import numpy
from spacy.en import English
import numpy as np
import argparse
import math
from tempfile import TemporaryFile
sys.path.append('/datadrive/ML/jasper/python/embedding-evaluation/wordsim/')
from wordsim import Wordsim
import data_wrangler

DEBUG = True


def text_to_embeddings_store(model_path, input_file, output_file, vector_dimension):
    """
    Takes an input_file and conby converts it to an output_file by replacing words with embedding vectors based on the model at model_path
    """
    # TODO: Refactor to call to text_to_embeddings
        #model_path = args.model_path_
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("Loafing spaCy model, this can take some time...")
    nlp=English()
    #print(("The vocabulary size is: "+str(len(model.vocab))))
    #print("Vector for 'Shubham': " + str(model['Shubham']))
    #print("Embedding dimension: " + str(len(model['Shubham'])))
    #f1=open("embedding_vectors_400.txt","w")
    f1=open(output_file,'w')
    zero =  np.zeros((vector_dimension,), dtype=np.float)
    #Specify encoding with io.open (careful io.open is slow, don't use for large files)
    #latin-1 is usually the culprit if the files aren't utf-8 encoded
    #with io.open("dataset_latin-1.txt", "r", encoding='latin-1') as f:
    count=0
    max_length=0
    with io.open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            #spaCy would do this better :)
            #row=line.split()
            doc = nlp(line)                
            arr = []
            #for i in range(0,len(doc)):
            for token in doc:
                try:
                    embedding = model[token.text]
                    #print("Success for:\t" + token.text)
                except KeyError:
                    #print("Fail for:\t" + token.text)
                    #TODO: set embedding to zero vector instead of continue
                    embedding = zero
                #temp=str(model[row[i]])
                #temp.replace('\n',' ')
                #f1.write(temp)
                arr.append(embedding)
                #TODO: write as one line using join method
                #f1.write(str(embedding))
                #f1.write(" ")
            rows,cols=np.shape(arr)
            if rows==0:                 #ignore the tweet if out of vocabulary and take the control to the beginning of the loop
                count=count+1
                continue
            temp = arr[0]
            if (rows>max_length): # maximum words in a sentence
                max_length=rows
            for i in range(1,rows):
                temp=np.concatenate((temp,arr[i]),axis=0)
            rand=' '.join(map(str,temp))
            f1.write(rand)
            f1.write("\n")
    print("There are"+str(count)+"out of vocabulary sentences.")
    print(max_length)   
    return max_length
    
    
def get_max_length(input_file, nlp=None):
    """
    Get maximal number of spaCy tokens in input_file. Feel free to pass the spaCy model if you already have it.
    """
    
    #load spacy for tokenization
    if nlp is None:
        nlp=English()

    max_length = 0
    with io.open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            #get text content from tsv column 2
            rows = line.replace("\n"," ").split("\t")
            text = rows[2]

            doc = nlp(text)
            length = doc.__len__()
            if length > max_length:
                max_length = length
                
    return max_length


def text_to_embeddings(input_file, model_path, word_vec_length, doc_length, n_docs=None, one_hot=True):
    """
    Takes an input_file and converts it to a npy output file by replacing words with embedding vectors.

    Embeddings are based on the model at model_path.
    Word embeddings vectors are concatenated to a document vector of doc_length embedding vectors. All vectors are
    padded to doc_length.

    The output file contains a 2D array where the 0th column are the ids, the 1st column are the labels and the rest
    are the document vector values.


    :param input_file: string
        Path to tsv input file (<id>\t<label>\t<text>) with one document per line
    :param model_path: string
        Path to embedding model file
    :param word_vec_length: int
        Dimension of embeddings in embedding model
    :param doc_length: int
        Maximal document length, doc embeddings will be trimmed/padded to this length
    :param one_hot: boolean
        Flag whether output labels should be one hot encoded (default: True)
    :return: numpy array [Number of lines in input file; 2 + word_vec_length * doc_length]
        2D array where the 0th column are the ids, 1st column are the labels and rest are the document vector values
    """
    if n_docs is None:
        n_docs = data_wrangler.get_n_lines(input_file)
    if DEBUG:
        print("n_docs: {0:d}".format(n_docs))

    print("Loading embedding model, this can take some time...")
    model = Wordsim.load_vector(model_path)
    print("Loading spaCy model, this can take some time...")
    nlp = English()

    max_doc_length = 0
    sum_total_words = 0
    doc_count = 0
    n_matches = 0

    id_list = []
    label_list = []
    doc_matrix = np.empty((n_docs, doc_length * word_vec_length))
    for data_item in data_wrangler.read_data_gen(input_file):
        if "id" in data_item:
            id = data_item["id"]
        else:
            id = doc_count
        label = data_item["label"]
        text = data_item["text"]

        # Log ids
        id_list.append(id)
        # log labels
        label_list.append(label)

        doc = nlp(text)
        length = doc.__len__()
        if length > max_doc_length:
            max_doc_length = length
        if length > doc_length:
            print("WARN: Number of tokens in current input doc ({0:d}) is larger than max length ({1:d}). Document"
                  "embedding will be trimmed to max length. Increase max length if this is not intended. Reported "
                  "embedding matches will be lowered by trimmed words."
                  .format(length, doc_length))
            print(text)

        sum_total_words += length

        doc_embedding, n_tokens_doc, n_matches_doc = doc_to_vec(model, doc, word_vec_length, doc_length)
        n_matches += n_matches_doc

        doc_matrix[doc_count, :] = doc_embedding
        doc_count += 1

    # Concatenate ids
    ids = np.asarray(id_list, dtype=np.float)
    # Ensure labels start with 0 and converted from str to int
    label_list = data_wrangler.encode_labels(label_list)
    # Concatenate labels
    labels = np.asarray(label_list, dtype=np.int)
    if one_hot:
        labels = data_wrangler.one_hot_conversion(labels)

    print("Matched {0:d}/{1:d} tokens, or {2:5.2f}%"
          .format(n_matches, sum_total_words, float(n_matches) / sum_total_words))
    print('Document trim/pad length: ' + str(doc_length))
    print('Maximal input doc length: ' + str(max_doc_length))
    print('Average sentence length: ' + str(sum_total_words / doc_count))

    if DEBUG:
        # Extra debugging info
        print("ids.shape: {}".format(ids.shape))
        print("labels.shape: {}".format(labels.shape))
        print("doc_matrix.shape: {}".format(doc_matrix.shape))
        print("DEBUG: IDS")
        print("DEBUG: " + str(ids[0:5]))
        print("DEBUG: LABELS")
        print("DEBUG: " + str(labels[0:5]))
        print("DEBUG: DOC ROW 0")
        print("DEBUG: " + str(doc_matrix[0, :]))

    return ids, labels, doc_matrix


def text_to_embeddings_npy(input_file, output_file, model_path, word_vec_length, doc_length):
    """
    Takes an input_file and converts it to a npy output file by replacing words with embedding vectors.

    Embeddings are based on the model at model_path.
    Word embeddings vectors are concatenated to a document vector of max_length embedding vectors. All vectors are
    padded to max_length.

    The output file contains a 2D array where the 0th column are the ids, the 1st column are the labels and the rest
    are the document vector values.

    :param model_path: string
        Path to embedding model file
    :param word_vec_length:
        Dimension of embeddings in embedding model
    :param input_file: string
        Path to tsv input file (<id>\t<label>\t<text>) with one document per line
    :param output_file: stinrg
        Path to npy output file (.npy will be concatenated if not present)
    :param doc_length:
        Maximal document length, doc embeddings will be trimmed/padded to this length
    :return: numpy array [Number of lines in input file; 2 + vector_dimension * max_length]
        2D array where the 0th column are the ids, 1st column are the labels and rest are the document vector values
    """

    ids, labels, doc_matrix = text_to_embeddings(
        input_file,
        model_path,
        word_vec_length,
        doc_length,
        one_hot=False
    )
    
    # Stack ids, labels and doc_embeddings into data matrix
    data = np.column_stack((ids, labels, doc_matrix))
    
    np.save(output_file, data)

    print("Stored ids, labels and doc embeddings for " + str(data.shape[0]) + " docs in " + output_file)

    return data


def text_to_vec(model, nlp, text, word_vec_length, doc_length):
    """
    Takes a text string and converts it do a document embedding.

    Unknown word embeddings are set to zero. Document embedding is zeroed or trimmed to max_length * vector_dim

    :param model: dictionary {string: array of floats}
        Word embedding model
    :param nlp: spaCy model
        spaCy model for tokenizing text
    :param text: string
        Input text
    :param word_vec_length: int
        Word embedding dimension
    :param doc_length: int
        Maximal number of words in document
    :return: array of float, shape=(max_length * vector_dim)
        Concatenation of word embeddings
    """
    doc = nlp(text)

    length = doc.__len__()
    if length > doc_length:
        print("WARN: Number of tokens in current input doc ({0:d}) is larger than max length ({1:d}). Document"
              "embedding will be trimmed to doc length. Increase max length if this is not intended."
              .format(len(embedd_list), doc_length))

    doc_embedding, n_tokens_doc, n_matches_doc = doc_to_vec(model, doc, word_vec_length, doc_length)

    return doc_embedding


def doc_to_vec(model, doc, word_vec_length, doc_length):
    """
    Takes a spaCy doc and converts it do a document embedding.

    Unknown word embeddings are set to zero. Document embedding is zeroed or trimmed to max_length * vector_dim

    :param model: dictionary {string: array of floats}
        Word embedding model
    :param doc: spaCy doc
        spaCy document object
    :param word_vec_length: int
        Dimension of embeddings in embedding model
    :param doc_length: int
        Maximal number of words in document
    :return: array of float, shape=(max_length * vector_dim)
        Concatenation of word embeddings
    """
    doc_embedding = np.zeros((word_vec_length * doc_length), dtype=np.float)
    n_tokens = 0
    n_matches = 0
    for token in doc:
        if n_tokens >= doc_length:
            break
        try:
            embedding = np.array(model[token.text], dtype=np.float)
            n_matches += 1
        except KeyError:
            n_tokens += 1
            # No need to set embedding to zero as doc embedding is initialized to zero -> continue
            continue
        start = n_tokens * word_vec_length
        end = (n_tokens + 1) * word_vec_length
        doc_embedding[start:end] = embedding
        n_tokens += 1

    return doc_embedding, n_tokens, n_matches


def load_embeddings_npy(input_file_path, padding_dimension=None):
    data = np.load(input_file_path)
    tweet_ids = data[:,0]

    y = np.array(data[:,1],dtype=int)
    x = data[:,2:]

    return x, y, tweet_ids
    

def main():
    """
    Run text_to_embeddings_npy based on argparse input.

    For example:
    python \
    text_to_embeddings.py \
    --model_path /datadrive/nlp/jasper/w2v/jinho/w2v-twitter-archiveteam-400.model \
    --input_file /datadrive/nlp/jasper/smm4h/smm4h_task2_dev.txt \
    --output_file /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_dev_jinho_400.npy \
    --vector_dimension 400 \
    --document_length 47

    See text_to_embedings_cmd.txt in same folder for more examples.

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the word2vec twitter model bin file')
    parser.add_argument('--input_file', help='File to read the input text data')
    parser.add_argument('--output_file', help='numpy file to write ids,labels and vectors')
    parser.add_argument('--document_length', '-d', type=int, default=120,
                        help="Number of input tokens per document (default: 120)")
    parser.add_argument('--vector_dimension', type=int, help='dimension of output vector using word2vec model')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg + ': ' + str(getattr(args, arg)))

    text_to_embeddings_npy(args.input_file, args.output_file, args.model_path, args.vector_dimension,
                           args.document_length)
    return None

if __name__ == "__main__":
    main()
