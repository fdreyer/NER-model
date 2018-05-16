#!/usr/bin/env python3
#
# Frederic Dreyer, 2018
#

"""This file contains all the named entity recognition models:
 - NaiveDicNER: a naive model that looks for previously identified labels.
 - CrfNER: a model based on Conditional Random Fields.
 - SpacyNER: a NER model using the spaCy library.
 - NER: a wrapper that can be set to use any of the above.
All are derived from a common NERModel class."""

import random, spacy, re, nltk, sklearn_crfsuite
from pathlib import Path
from abc import ABC, abstractmethod


class NERModel(ABC):
    """An abstract base class for our NER models."""
    
    def __init__(self, label):
        self.model = None
        self.label = label        
        
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    def score(self, valid_data, beta=1.0):
        """Score the trained model against a validation set and return the F-measure"""
        # first make sure that the model has been trained
        if not self.model:
            print("ERROR: Model needs to be trained first.")
            return
        
        # annotate the sentences from the validation data and count
        # the true positives, false positives, and false negatives
        true_pos_sum  = 0
        false_pos_sum = 0
        false_neg_sum = 0
        for line, annot_true, _ in valid_data:
            annot_model = self._evaluate(line.rstrip())
            true_pos  = [a for a in annot_model['entities'] if a in annot_true['entities']]
            true_pos_sum  += len(true_pos)
            false_neg_sum += len(annot_true['entities'])  - len(true_pos)
            false_pos_sum += len(annot_model['entities']) - len(true_pos)
            # less efficient implemention
            # true_pos  = [a for a in annot_model['entities'] if a in annot_true['entities']]
            # false_pos = [a for a in annot_model['entities'] if a not in annot_true['entities']]
            # false_neg = [a for a in annot_true['entities'] if a not in annot_model['entities']]
            # true_pos_sum  += len(true_pos)
            # false_pos_sum += len(false_pos)
            # false_neg_sum += len(false_neg)
            
        # calculate the F_beta score (default is F1 measure with beta=1)
        div   = (1.0 + beta**2) * true_pos_sum + beta**2 * false_neg_sum + false_pos_sum
        score = (1.0 + beta**2) * true_pos_sum
        if (div>0.0):
            score = score / div
        recall = 0.0
        precis = 0.0
        if (true_pos_sum + false_pos_sum > 0.0):
            precis = true_pos_sum / (true_pos_sum + false_pos_sum)
        if (true_pos_sum + false_neg_sum > 0.0):
            recall = true_pos_sum / (true_pos_sum + false_neg_sum)
        print('F-measure:{:8.5f}\nprecision:{:8.5f}\nrecall:{:11.5f}'.format(score, precis, recall))
        return score
    
    def annotate(self, textfile):
        """Annotate a given text file using the trained NER model."""
        # first make sure that the model has been trained
        if not self.model:
            print("ERROR: Model needs to be trained first.")
            return
        
        # loop over each line and annotate them
        result = []
        for nline, line in enumerate(open(textfile,'r')):
            annotation = self._evaluate(line.rstrip())
            result.append((line, annotation, nline))
        return result

    def load(self, path):
        pass

    
class NaiveDicNER(NERModel):
    """Naive NER model using just a dictionary. This model is both slow and ineffective."""
    
    def __init__(self, label):
        NERModel.__init__(self, label)
        self.name = 'Naive Dictionary NER'
        
    def train(self, train_data):
        """Train the model by creating a list with all known entities."""
        print('Creating a list of entities with the',self.name,
              'model using',self.label,'labels.')
        entities = set()
        for line, annotations, _ in train_data:
            for ind_start, ind_end, _ in annotations['entities']:
                entity = line[ind_start:ind_end]
                # dirty hack to avoid regex issue later on:
                # we replace '(' and ')' with '\(' and '\)'
                entity = entity.replace('(','\(')
                entity = entity.replace(')','\)')
                # add entity to set if it is more then 3 characters
                # (otherwise it blows up the false positive count)
                if len(entity)>2:
                    entities.add(entity)
        # now save the set of entities as a list
        self.model = list(entities)
        # and order it in the length of the strings
        self.model.sort(key = lambda s: -len(s))

    def _evaluate(self, line):
        """Find all known entities occurring in the line."""
        # make sure the model has be "trained"
        if not self.model:
            print("ERROR: Model needs to be trained first.")
            return

        # set up the annot dictionary and the list of indices
        annot = {'entities':[]}
        indices = []
        # loop over all entities in our list
        for entity in self.model:
            # find all matching patterns in the line
            for match in re.finditer(entity, line):
                istart = match.start()
                iend   = match.end()
                # if the match is not already covered, add it to the
                # annotations
                if not self._overlapping(istart, iend, indices):
                    indices.append((istart, iend))
                    annot['entities'].append((istart, iend, self.label.upper()))
        return annot
            
    def _overlapping(self, i1, i2, indices):
        """Check if the indices i1, i2 are already in the covered range"""
        overlap = False
        for j1, j2 in indices:
            # true if [i1, i2) overlaps with [j1, j2)
            if not overlap:
                overlap = i1 < j2 and j1 < i2
        return overlap


class CrfNER(NERModel):
    """NER model using a Conditional Random Field."""
    # this model is based on
    # https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
    
    def __init__(self, label):
        NERModel.__init__(self, label)
        self.name = 'CRF NER'
    
    def train(self, training_data, n_iter = 100):
        """Train the CRF on input data."""
        print('Beginning the training of the',self.name,'model with',self.label,'labels.')
        # convert to useful format
        training_data = self._convert(training_data)
        # create features and labels
        X_train = [self._sent2features(s) for s in training_data]
        y_train = [[label for token, postag, label in s] for s in training_data]

        # now set up the CRF model
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.05, c2=0.05, 
                                   max_iterations=n_iter)
        crf.fit(X_train, y_train)
        self.model = crf


    def _evaluate(self, line):
        """Evaluate a line with the trained model and return annotations."""
        if not self.model:
            print("ERROR: Model needs to be trained first.")
            return
        annot  = []
        if len(line)==0:
            return {'entities':[]}
        tagged = nltk.pos_tag(nltk.word_tokenize(line))
        features = self._sent2features(tagged)
        result = self.model.predict([features])[0]
        index = 0
        while(index < len(line)):
            for elem, pred in zip(tagged, result):
                index = line.find(elem[0],index)
                if pred!='O':
                    annot.append([index, index + len(elem[0]),
                                              pred])
                index += len(elem[0])
            break
        # now combine beginning and internal accordingly
        for i in range(len(annot)-1,-1,-1):
            if (i>0) and (annot[i][2]=='I-'+self.label.upper()):
                annot[i-1][1] = annot[i][1]
                del annot[i]
            elif (annot[i][2]=='B-'+self.label.upper()):
                annot[i][2]=annot[i][2][2:]
        return {'entities': [(i,j,t) for i,j,t in annot]}
                

    def _convert(self, data):
        """Change from the spaCy format to something parseable by our CRF model."""
        # NOTE: this is very hacky, and should be cleaned up at some point
        # tag should be something not present in the text
        tag='mgcTGPulbVyalXSoHJYtJYFM6SqSTC' # 30 randomly generated characters 
        result = []
        for line, entities, _ in data:
            entities = entities['entities']
            for i in range(len(line)-1, -1, -1):
                if i in [x[1] for x in entities]:
                    j = [x[0] for x in entities if x[1]==i][0]
                    line = line[:i]+tag+line[i:]
                    for k in range(i-1,j-1,-1):
                        if line[k]==' ':
                            line=line[:k]+tag+line[k:]
            line = nltk.pos_tag(nltk.word_tokenize(line))
            # add the names, and remove the tag that was added for bookkeeping
            res = [[x[0],x[1],'O'] if tag not in x[0] else \
                   [x[0].replace(tag,''),x[1],self.label.upper()] for x in line]

            # append the B and I tags 
            if len(res)>0:
                if res[0][2]==self.label.upper():
                    res[0][2]='B-'+res[0][2]
            for i in range(1,len(res)):
                if ((res[i-1][2] == 'B-'+self.label.upper()) or\
                    (res[i-1][2] == 'I-'+self.label.upper())):
                    if res[i][2] == self.label.upper():
                        res[i][2] = 'I-'+res[i][2]
                elif (res[i][2] == self.label.upper()):
                    res[i][2] = 'B-'+res[i][2]
            result.append([tuple(x) for x in res])
        return result
            
    # taken from https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
    def _word2features(self, sent, i):
        """Return a dictionary of features from the element given as input."""
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],        
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
                    
        return features
    
    # taken from https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
    def _sent2features(self, sent):
        """Transform the input data into usable features"""
        return [self._word2features(sent, i) for i in range(len(sent))]
    
    
class SpacyNER(NERModel):
    """NER model using spaCy."""
    
    def __init__(self, label):
        NERModel.__init__(self, label)
        self.name = 'spaCy NER'
    
    def train(self, train_data, n_iter = 20, outdir='model/'):
        """Set up an NER model and train it on input data, then save the model to disk."""
        
        # create blank Language class and add entity recognizer with label
        nlp = spacy.blank('en')
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)    
        ner.add_label(self.label.upper()) 
        
        # start the training
        print('Beginning the training of the',self.name,'model with',self.label,'labels.')
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            status = {}
            for txt, annotations, _ in train_data:
                nlp.update([txt], [annotations], sgd=optimizer,
                           drop=0.3, losses=status)
            print('Iteration %i/%i:'%(itn+1,n_iter),status)
        
        # update the internal model
        self.model = nlp
        
        # save model to output directory
        outdir = Path(outdir)
        if not outdir.exists():
            # create directory if it does not exist already
            print("Creating directory", outdir,"and writing out model.")
            outdir.mkdir()
        else:
            # otherwise overwrite the directy with current model
            print("WARNING: overwriting the",outdir,"directory.")
        nlp.meta['name'] = self.label
        nlp.to_disk(outdir)
        
    def load(self, path='model/'):
        """Load in a previously trained model"""
        path = Path(path)
        self.model = spacy.load(path)

    def _evaluate(self, line):
        """Apply the NER model to input line and return labels"""
        doc = self.model(line)
        annot = {'entities':[]}
        for ent in doc.ents:
            annot['entities'].append((ent.start_char, ent.end_char, ent.label_))
        return annot


# # TODO
# class LstmNER(NERModel):    
#     """NER model using an LSTM approach."""
#
#     def __init__(self, label):
#         NERModel.__init__(self, label)
#         self.name = 'LSTM NER'
#   
#     def train(self, train_data):
#         pass
#
#     def evaluate(self, line):
#         if not self.model:
#             print("ERROR: Model needs to be trained first.")
#             return
#         pass


class NER:
    """Wrapper for NER models."""
    
    def __init__(self, model='spacy', label):
        if model == 'spacy':
            self.model = SpacyNER(label)
        elif model == 'crf':
            self.model = CrfNER(label)
        elif model == 'dic':
            self.model = NaiveDicNER(label)
        # elif model == 'lstm':
        #     self.model = LstmNER(label)
        else:
            raise ValueError("NER model must be: spacy, crf or dic")
    
    def train(self, training_data):
        """Train the model on input data."""
        self.model.train(training_data)

    def score(self, valid_data, beta=1.0):
        """Score the model on validation data."""
        return self.model.score(valid_data, beta)
        
    def annotate(self, textfile):
        """Annotate a given list of sentences using the trained NER model."""
        return self.model.annotate(textfile)
    
    def load(self, path='model/'):
        if not (self.model.name=='spaCy NER'):
            raise ValueError("Only spacy model can be loaded from file")
        self.model.load(path)
    
