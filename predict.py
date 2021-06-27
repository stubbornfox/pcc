import sys
from PIL import Image
import urllib
from config import dataset
from utils.graph.decision_path import draw_decision_path


if __name__ == "__main__":
  for arg in sys.argv[1:]:
    bird = str(arg)
    if bird.isdigit():
      bird = int(bird)
    else:
      bird = Image.open(urllib.request.urlopen(bird))

    draw_decision_path(bird, dataset)
