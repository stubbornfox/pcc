from utils.graph.predict import guess_bird
from config import configuration, dataset
import  numpy as np
from utils.visualization import view_source_image,view_segments_of, view_concepts_of
from steps.s4_discover_concepts import load_concepts, global_index_mapping
from steps.s5_build_decision_tree import predict_bird


from sklearn.metrics import accuracy_score
train_ids, test_ids = dataset.train_test_image_ids()
Y_train, Y_test = dataset.train_test_class_ids()
np.random.seed(42)
predict_bird(test_ids[150:151], configuration)
print(Y_test[150])
# test_ids = np.random.choice(test_ids, 5)
# index_mapping = global_index_mapping(train_ids)
# a = []
# classes_per_image_id = dataset.classes_per_image_id(True, True)
# print(accuracy_scoreccuracy_score(predict_bird(test_ids, configuration), Y_test))
# for id in test_ids[:10]:
#   print(id, classes_per_image_id[id])
#   guesses = guess_bird(id, dataset, index_mapping, classes_per_image_id)
#   a.append(guesses)
# print(a)
# print(Y_test)
# print(accuracy_score(a, Y_test))
  # print(id, classes_per_image_id[id], guesses)
  # for gu in guesses:
  #   print(classes_per_image_id[gu])

    # print(id)
    # view_source_image(id)
    # view_segments_of(id)