from util import *
from discover_concepts import _concepts, _important_concepts
import numpy as np
from sklearn import tree as tree
from sklearn import ensemble
from  sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import math
from sklearn.cluster import AgglomerativeClustering
# from IPython.display import Image

def calculate_vector_to_concepts(acts_of_img, concepts, kmeanmodel):
  output = []
  for concept in concepts:
    _, _, center, cluster_lbl = concept
    min_distance = 500
    for act in acts_of_img:
      distance = euclidean_distance(act, center)
      if distance < min_distance:
        min_distance = distance
    # min_distance = int(math.floor(min_distance) <= 6)
    output.append(min_distance)
  return np.array(output)

def calculate_has_concepts(acts_of_img, concepts, kmeanmodel):
  output = []
  centers, cluster_lbls = [], []
  for concept in concepts:
    _, _, center, cluster_lbl = concept
    centers.append(center)
    cluster_lbls.append(cluster_lbl)
  predict = kmeanmodel.predict(acts_of_img)
  kmean_center = kmeanmodel.cluster_centers_
  keep = []
  for (act, cl) in zip(acts_of_img, predict):
    a = euclidean_distance(act, kmean_center[cl])
    if a <= 40:
      keep.append(cl)

  for lbl in cluster_lbls:
    if lbl in keep:
      output.append(1)
    else:
      output.append(0)
  print(output)
  return np.array(output)

def build_attributes(images_acts, concepts, kmeanmodel):
  output = []
  for acts_of_img in images_acts:
    # output.append(calculate_has_concepts(acts_of_img, concepts, kmeanmodel))
    output.append(calculate_vector_to_concepts(acts_of_img, concepts, kmeanmodel))

  return np.array(output)

def concept_treeeee():
  train_ids = load_images(True)
  test_ids = load_images(False)
  y_train = build_labels_data(train_ids)
  y_test = build_labels_data(test_ids)
  concepts = _important_concepts()
  train_images_acts = load_img_activation(train_ids)
  test_images_acts = load_img_activation(test_ids)
  kmeanmodel = load_kmean_model()
  print('Numer of concepts:', len(concepts))
  X_train = build_attributes(train_images_acts, concepts, kmeanmodel)
  X_test = build_attributes(test_images_acts, concepts, kmeanmodel)

  # optimized_tree(X_train, y_train, X_test, y_test)

  clf = ensemble.RandomForestClassifier(min_samples_leaf=5, min_samples_split=5, max_features='auto', n_estimators=10)
  clf = clf.fit(X_train, y_train)

  print("Accuracy Score on Train:")
  print(clf.score(X_train, y_train, sample_weight=None))
  print("Accuracy Score on Test:")
  print(clf.score(X_test, y_test, sample_weight=None))
  x_path = "result_fn_{}_birds.csv".format(NUMBER_CLASS)
  a = clf.predict(X_test)
  y_test = np.array(y_test)
  a = np.array(a)
  print('Accuracy on Test', accuracy_score(y_test, a))
  np.savetxt(x_path, np.array(confusion_matrix(y_test, a)).astype(int), fmt='%5.0f')
  estimator = clf.estimators_[5]

  # from sklearn.tree import export_graphviz
  # # Export as dot file
  # export_graphviz(estimator, out_file='tree.dot',
  #                 feature_names=iris.feature_names,
  #                 class_names=iris.target_names,
  #                 rounded=True, proportion=False,
  #                 precision=2, filled=True)
  #
  # # Convert to png using system command (requires Graphviz)
  # from subprocess import call
  # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
  #
  # # Display in jupyter notebook
  #
  # Image(filename='tree.png')

  return clf, estimator, concepts, kmeanmodel

def evaluate(model, test_features, test_labels):
    print(model.score(test_features, test_labels, sample_weight=None))
    # predictions = model.predict(test_features)
    # errors = abs(predictions - test_labels)
    # mape = 100 * np.mean(errors / test_labels)
    # accuracy = 100 - mape
    # print('Model Performance')
    # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    # print('Accuracy = {:0.2f}%.'.format(accuracy))
    #
    # return accuracy

def optimized_tree(X_train, y_train, X_test, y_test):
  clf = ensemble.RandomForestClassifier()
  from sklearn.model_selection import RandomizedSearchCV
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start=100, stop=200, num=10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(150, 220, num=11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [10, 13, 15]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [8, 10, 15]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}
  rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                 random_state=42, n_jobs=-1)
  rf_random.fit(X_train, y_train)
  best_random = rf_random.best_estimator_
  print(rf_random.best_params_)
  print('Train Accuracy')
  random_accuracy = evaluate(best_random, X_train, y_train)
  print('Test Accuracy')
  random_accuracy = evaluate(best_random, X_test, y_test)

def print_out_concepts(cl, nodes, posx, posy):
  img_ids = load_images(True)
  concept_locs = locate_concepts_ids(img_ids)
  b = []
  for i in cl:
    _, ids, _, _ = i

    a = load_segments(True, ids[:1], concept_locs)
    b.append(a)

  index = 0

  fig, ax = plt.subplots(len(posx), len(posy), figsize=(5, 5), sharex=True, sharey=True)

  for segement_t in b:
    segement = segement_t[0]
    x, y = posx[index], posy[index]
    image = Image.fromarray((segement * 255).astype(np.uint8))
    ax[x, y].imshow(image)
    index += 1

  for a in ax.ravel():
    a.set_axis_off()

  plt.tight_layout()
  plt.show()

def plot_decision_path(img_id):
  model, clf, concepts, kmeanmodel = concept_treeeee()
  # ccc=[]
  # for cc in concepts:
  #   _, ids, _, _ = cc
  #   print(ids)
  #   ccc.append(ids)
  img_ids = [img_id]
  acts = load_img_activation(img_ids)
  X_test = build_attributes(acts, concepts, kmeanmodel)

  feature = clf.tree_.feature
  print('Feature Lenght', len(feature))
  print('Concept Lenght', len(concepts))
  threshold = clf.tree_.threshold

  node_indicator = clf.decision_path(X_test)
  print('Predict label', model.predict(X_test))
  leaf_id = clf.apply(X_test)

  sample_id = 0
  # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
  node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                      node_indicator.indptr[sample_id + 1]]

  print('Rules used to predict sample {id}:\n'.format(id=sample_id))
  nodeids = []
  pos_x, pos_y = [], []
  count = 0
  y = 0
  for node_id in node_index:
    # continue to the next node if it is a leaf node
    if leaf_id[sample_id] == node_id:
      continue
    pos_x.append(count)
    pos_y.append(y)
    count += 1
    nodeids.append(feature[node_id])
    # check if value of the split feature for sample 0 is below threshold
    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
      threshold_sign = "<="
      y -= 1
    else:
      threshold_sign = ">"
      y += 1


    print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
          "{inequality} {threshold})".format(
      node=node_id,
      sample=sample_id,
      feature=feature[node_id],
      value=X_test[sample_id, feature[node_id]],
      inequality=threshold_sign,
      threshold=threshold[node_id]))

    # print(ccc[node_id])
  print(pos_x, pos_y)
  pos_x =np.array(pos_x)
  pos_y =np.array(pos_y)
  miny=np.min(pos_y)
  pos_y = pos_y + miny
  print(pos_x, pos_y)
  print(nodeids)
  print('number of nodes',len(concepts[nodeids]))
  print_out_concepts(concepts[nodeids], len(nodeids), pos_x, pos_y)
# concept_treeeee()
plot_decision_path(631)