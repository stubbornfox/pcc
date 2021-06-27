# ProtoTree Construction with Cluster Analysis

## Setup

`pip install -r requirements.txt`

The project was tested under Macos using Python 3.7 and 3.9. Due to hardware 
limitations the training was done _without_ cuda support enabled, so on the CPU.

## Structure

The main files to interact with are `train.py` and `predict.py`. Running 

```shell
python3 train.py
```

will train a decision tree model and write it (and a lot of other files) to 
disk. `predict.py` is a CLI script that loads the pre-trained model from disk 
and visualizes the classification result.

The high level steps needed for the training are listed and described in 
`train.py`. Each step has it's in the `steps/` directory, where one can see more 
detailed actions.

All supporting code lives in the `utils/` directory. The configuration values 
can be adjusted in `config.py` and are explained in more detail in 
`utils/configuration.py`.

## Data

In order for the code to work, the dataset has to be placed in the 
`data/CUB_200_2011` folder. All files generated by `train.py` will also live 
inside that folder. 

## Prediction

After successfully running `train.py`, the prediction can be done as follows:

- id: `run python predict.py 1`, where one provides a valid image id, which can 
  be obtained by looking at `data/CUB_200_2011/images.txt`.

- url: `run python predict.py "https://live.staticflickr.com/65535/40705400433_d416c53cee_k.jpg"`, where the
  url points to any image file that containing a bird out of the 200 classes 
  that the model was trained upon. 

After the classification is done, open your browser at `http://127.0.0.1:8050/` 
to see the graph containing the explanation.
