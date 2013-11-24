import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as pyplot

def import_data(input_data):
  """
  Convert the raw input using the following specification
  first line- N, M (Number of training records, Number of parameters)
  (N lines of data), <ID> <+1/-1> (<F-ID>:<value>)*
  q (number of records to be classified)
  (q lines of data), <ID> (F-ID>:<value>)*
  """
  training_set = []
  pred_set = []
  M, N, q = (0, 0, 0)
  for i, l in enumerate(input_data):
    split_row = l.split(" ")        
    #if first row, then grab M and N
    if i==0:
      N, M = map(lambda s: int(s), split_row)
    #if one of the N rows following, process with the classification label
    elif i==N+1:
      q = int(split_row[0])
    elif (1<= i <=N) or (N+2<= i <=(N+2+q)):
      attributes = {}

      if 1<= i <=N: non_features = ["ID", "Label"]
      else: non_features = ["ID"]

      varnames = non_features + ["F" + str(j) for j in xrange(1, M+1)]
      
      for j, v in enumerate(varnames):
        if v in non_features: attributes[v] = split_row[j].rstrip()
        else:
          #strip out the number, and use that to construct the next attribute
          f_id, f_value = split_row[j].split(":")
          if "F" + str(f_id.rstrip()) != v: raise NameError('feature_id out of order')
          attributes[v] = float(f_value)

      if 1<= i <=N:  
        training_set.append(attributes)
      else:
        pred_set.append(attributes)
  
  #check that the sets are properly constructed
  if len(training_set) != N: raise NameError('training set smaller than expected')
  elif len(training_set[0]) != 2 + M: raise NameError('fewer variables in training set than expected')
  elif len(pred_set) != q: raise NameError('less than q records to be classified')
  elif len(pred_set[0]) != 1 + M: raise NameError('fewer variables in to-classify set than expected')
  
  return (M, N, q, training_set, pred_set)

def import_cv_results(input_data):
  """
  Cross validation dataset has been provided to allow for fine-tuning of the learning algorithm.
  """
  cv_results = []
  for l in input_data:
    split_row = l.split(" ")
    cv_results.append({"ID": split_row[0].rstrip(), "Label": split_row[1].rstrip()})

  return cv_results                       

def main():
  """
  Use random forests to classify, based on cv results
  """
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  from sklearn.grid_search import GridSearchCV
  from sklearn.preprocessing import StandardScaler
  import sys 

  #call import_data on STDIN, returning formatted query results
  M, N, q, tset, pset = import_data(sys.stdin)
  #create features list so we can easily grab the feature fields
  features = ["F" + str(j) for j in xrange(1, M+1)]
 
  #read in to a pandas dataframe and perform some preprocessing
  training_set = pd.DataFrame(tset).set_index('ID')
  pred_set = pd.DataFrame(pset).set_index('ID')
  
  scale = StandardScaler().fit(training_set[features])
  training_set[features] = scale.transform(training_set[features])
  pred_set[features] = scale.transform(pred_set[features])
  
  #adjust the labeling convention
  training_set['Label'] = training_set['Label'] == "+1"

  grad = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1)
  grad.fit(training_set[features], training_set['Label'])

  def print_results(x):
    if x['Pred_Label'] == 1: print x.name + " +1"
    else: print x.name + " -1" 
    
  pred_set['Pred_Label'] = grad.predict(pred_set[features])
  pred_set.apply(print_results, axis=1)

if __name__ == '__main__':
  main()
