
# Comparing Meta-Classifiers for Automatic Music Genre Classification

Source code from proceding Comparing Meta-Classifiers for Automatic Music Genre Classification published at 17th Brazilian Symposium on Computer Music (SBCM).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for experiment reproducibility.


### Prerequisites

- Python 3.x
- Libraries listed [here](https://github.com/vitorys/MusicGenreMetaClassifier) 

### Installing

A step by step series of examples that tell you how to get everything set up to run the experiment.

1. The first step is clone this repository.

```
$ git clone https://github.com/vitorys/MusicGenreMetaClassifier.git
```

2.  (Optional) Create a virtual enviroment and activate it.

```
$ virtualenv venv && activate venv/bin/activate
```

3. Install the requirements.

```
$ pip install -r requirements.txt
```


## Running experiments

### To run the Neural Network experiments:

1. Navigate to NeuralNetwork folder.
2. Execute the python file `main.py` follow by some dataset. For example:
```
python main.py data/gtzan-ds_rp-feats_frames
```
3. The result will be stored at ``output/`` folder.

### To run the Hidden Markov experiments:


1. Navigate to ``HMM`` folder. 
2. Execute the python file `main.py` follow by ``--input`` and some dataset. For example:
```
python classifier.py --input data/gtzan-ds_rp-feats_frames
```
4. The result will be stored at ``output/`` folder.


## Authors

* **Vítor Yudi Shinohara** - State University of Campinas
* **Juliano Henrique Foleiss** - The Federal University of Technology - Paraná
* **Tiago Fernandes Tavares** - State University of Campinas

