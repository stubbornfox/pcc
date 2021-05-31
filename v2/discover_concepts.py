import sklearn.cluster as cluster
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import pickle
from util import load_images, load_activations, load_cluster, \
      load_train_test, show_image, load_segments, load_concept, \
      euclidean_distance, locate_concepts_ids, load_concept_accuracy, \
      NUMBER_CLUSTER, NUMBER_CLASS, ACCURACY_SEGMENTS

def _cluster(acts):
      cluster_path = "v2/data/clusters/{}_{}.npz".format(NUMBER_CLUSTER, NUMBER_CLASS)

      if not os.path.isdir("v2/data/clusters"):
            os.makedirs("v2/data/clusters")

      if not os.path.exists(cluster_path):
            km = cluster.KMeans(NUMBER_CLUSTER)
            d = km.fit(acts)
            centers = km.cluster_centers_
            mc2 = []
            cc = km.cluster_centers_
            for act, lbl in zip(acts, km.labels_):
                  mc2.append(euclidean_distance(act, cc[lbl]))
            mc2 = np.array(mc2)
            asg = km.labels_
            cost = mc2
            np.savez_compressed(cluster_path, asg=asg, cost=cost, centers=centers)
            with open("v2/data/clusters/kmean_{}_{}.pkl".format(NUMBER_CLUSTER, NUMBER_CLASS), "wb") as f:
                  pickle.dump(km, f)

      return load_cluster(cluster_path)

def _concepts():
      concept_path = "v2/data/concept/{}_{}.npz".format(NUMBER_CLUSTER, NUMBER_CLASS)

      if not os.path.isdir("v2/data/concept"):
            os.makedirs("v2/data/concept")

      if os.path.exists(concept_path):
            return load_concept(concept_path)

      bn_dic = {}
      min_imgs = 20
      max_imgs = 40
      train_images, _ = load_train_test()
      bn_activations, image_numbers = load_activations()
      print(len(bn_activations))
      labels, cost, centers = _cluster(bn_activations)
      concept_number, bn_dic['concepts'] = 0, []

      bn_dic['label'] = labels
      bn_dic['cost'] = cost
      bn_dic['centers'] = centers
      print(len(bn_dic['label']))
      print('-----')
      print('Finish Cluster')
      concept_output = []
      for i in range(bn_dic['label'].max() + 1):
            label_idxs = np.where(bn_dic['label'] == i)[0]
            if len(label_idxs) > min_imgs:
                  concept_costs = bn_dic['cost'][label_idxs]
                  concept_idxs = label_idxs[np.argsort(concept_costs)[:max_imgs]]
                  concept_image_numbers = set(image_numbers[label_idxs])
                  discovery_size = len(train_images)
                  highly_common_concept = len(
                        concept_image_numbers) > 0.5 * len(label_idxs)
                  mildly_common_concept = len(
                        concept_image_numbers) > 0.25 * len(label_idxs)
                  mildly_populated_concept = len(
                        concept_image_numbers) > 0.25 * discovery_size
                  cond2 = mildly_populated_concept and mildly_common_concept
                  non_common_concept = len(
                        concept_image_numbers) > 0.1 * len(label_idxs)
                  highly_populated_concept = len(
                        concept_image_numbers) > 0.5 * discovery_size
                  cond3 = non_common_concept and highly_populated_concept
                  if highly_common_concept or cond2 or cond3:
                        concept_number += 1
                        tuple_concept = (concept_number, concept_idxs, centers[i], i)
                        concept_output.append(tuple_concept)
                        # concept = '{}_concept{}'.format('birds', concept_number)
                        # bn_dic['concepts'].append(concept)
                        # bn_dic[concept] = {
                        #       # 'images': self.dataset[concept_idxs],
                        #       # 'patches': self.patches[concept_idxs],
                        #       'image_numbers': image_numbers[concept_idxs]
                        # }
                        # segments = load_segments(True, concept_idxs)
                        #
                        # for sgm in segments[:5]:
                        #       show_image(sgm)
                        # return
                        # print(image_numbers[concept_idxs])
                        # bn_dic[concept + '_center'] = centers[i]

      print(concept_number)
      np.savez_compressed(concept_path, arr=concept_output)
      print(len(concept_output))
      return load_concept(concept_path)

def _important_concepts():
      concept_path = "v2/data/important_concept/{}_{}_{}.npz".format(NUMBER_CLUSTER, NUMBER_CLASS, ACCURACY_SEGMENTS)

      if  not os.path.isdir("v2/data/important_concept"):
            os.makedirs("v2/data/important_concept")

      if False and os.path.exists(concept_path):
            return load_concept(concept_path)

      bn_dic = {}
      min_imgs = 10
      max_imgs = 40
      train_images, _ = load_train_test()
      bn_activations, image_numbers = load_activations()
      print(len(bn_activations))
      local_concepts = locate_concepts_ids(load_images())
      labels, cost, centers = _cluster(bn_activations)
      concept_number, bn_dic['concepts'] = 0, []

      bn_dic['label'] = labels
      bn_dic['cost'] = cost
      bn_dic['centers'] = centers
      print(len(bn_dic['label']))
      print('-----')
      print('Finish Cluster')
      concept_output = []
      for i in range(bn_dic['label'].max() + 1):
            label_idxs = np.where(bn_dic['label'] == i)[0]
            if len(label_idxs) > min_imgs:
                  concept_costs = bn_dic['cost'][label_idxs]
                  concept_idxs = label_idxs[np.argsort(concept_costs)[:max_imgs]]
                  concept_image_numbers = set(image_numbers[label_idxs])
                  discovery_size = len(train_images)
                  highly_common_concept = len(
                        concept_image_numbers) > 0.5 * len(label_idxs)
                  mildly_common_concept = len(
                        concept_image_numbers) > 0.25 * len(label_idxs)
                  mildly_populated_concept = len(
                        concept_image_numbers) > 0.25 * discovery_size
                  cond2 = mildly_populated_concept and mildly_common_concept
                  non_common_concept = len(
                        concept_image_numbers) > 0.1 * len(label_idxs)
                  highly_populated_concept = len(
                        concept_image_numbers) > 0.5 * discovery_size
                  cond3 = non_common_concept and highly_populated_concept
                  if highly_common_concept or cond2 or cond3 or True:
                        if load_concept_accuracy(local_concepts, concept_idxs):
                              concept_number += 1
                              tuple_concept = (concept_number, concept_idxs, centers[i], i)
                              concept_output.append(tuple_concept)
                        # concept = '{}_concept{}'.format('birds', concept_number)
                        # bn_dic['concepts'].append(concept)
                        # bn_dic[concept] = {
                        #       # 'images': self.dataset[concept_idxs],
                        #       # 'patches': self.patches[concept_idxs],
                        #       'image_numbers': image_numbers[concept_idxs]
                        # }
                        # segments = load_segments(True, concept_idxs)
                        #
                        # for sgm in segments[:5]:
                        #       show_image(sgm)
                        # return
                        # print(image_numbers[concept_idxs])
                        # bn_dic[concept + '_center'] = centers[i]

      print('concept_number',concept_number)
      np.savez_compressed(concept_path, arr=concept_output)
      print(len(concept_output))
      return load_concept(concept_path)

_important_concepts()