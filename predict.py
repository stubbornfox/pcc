import sys
from PIL import Image
import urllib
from config import dataset
from decision_path import draw_decision_path


if __name__ == "__main__":
  bird = str(sys.argv[1])
  if bird.isdigit():
    bird = int(bird)
  else:
    bird = Image.open(urllib.request.urlopen(bird))

  draw_decision_path(bird, dataset)
