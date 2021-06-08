# ProtoTree Construction with Cluster Analysis

## Setup

`pip install -r requirements.txt`

The project was tested under Macos using Python 2.7 and 2.9. Due to hardware 
limitations the training was done _without_ cuda support enabled, so on the CPU.

## Structure

The main files to interact with are `train.py` and `classify.py`. Running 

```shell
python3 train.py
```

will train a decision tree model and write it (and a lot of other files) to 
disk. `classify.py` is a CLI script that loads the pre-trained model from disk 
and

The high level steps needed for the training are listed and described in 
`train.py`. Each step has it's in the `steps/` directory, where one can see more 
detailed actions.

All supporting code lives in the `utils/` directory.