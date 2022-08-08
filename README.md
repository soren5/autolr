# AutoLR: An evolutionary framework for learning rate optimizers
AutoLR is a framework based on the [SGE](https://github.com/nunolourenco/sge3 "SGE") engine capable of evolving learning rate optimizers for specific neural network architectures and problems. This repository is part of a published work; if you end up using this framework it would be appreciated that you reference one of the following:
```
@inproceedings{carvalho2020,
author = {Carvalho, Pedro and Louren\c{c}o, Nuno and Assun\c{c}\~{a}o, Filipe and Machado, Penousal},
title = {AutoLR: An Evolutionary Approach to Learning Rate Policies},
year = {2020},
isbn = {9781450371285},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3377930.3390158},
doi = {10.1145/3377930.3390158},
pages = {672–680},
numpages = {9},
keywords = {learning rate schedulers, structured grammatical evolution},
location = {Canc\'{u}n, Mexico},
series = {GECCO '20}
}
```
```
@inproceedings{carvalho2022, 
author=“Carvalho, Pedro and Louren\c{c}o, Nuno and Machado, Penousal”, 
editor=“Medvet, Eric and Pappa, Gisele and Xue, Bing”, 
title=“Evolving Adaptive Neural Network Optimizers for Image Classification”, 
booktitle=“Genetic Programming”, 
year=“2022”, 
publisher=“Springer International Publishing”, 
address=“Cham”, 
pages=“3–18”, 
abstract=“The evolution of hardware has enabled Artificial Neural Networks to become a staple solution to many modern Artificial Intelligence problems such as natural language processing and computer vision. The neural network’s effectiveness is highly dependent on the optimizer used during training, which motivated significant research into the design of neural network optimizers. Current research focuses on creating optimizers that perform well across different topologies and network types. While there is evidence that it is desirable to fine-tune optimizer parameters for specific networks, the benefits of designing optimizers specialized for single networks remain mostly unexplored.”, 
isbn=“978-3-031-02056-8” }
```

This system utilizes a grammar to generate a population of potential optimizers. The quality of these optimizers is then assessed using an evaluator. The best optimizers are then combined amongst themselves and altered in order to create a new population.

## Quickstart
This section will briefly explain how to run a simple AutoLR experiment.
The repository includes a `requirements.txt` file that should be used to install all necessary dependencies using the command:

```bash
> pip install -r requirements.txt
```

The user can use the Dockerfile to setup a container for the framework.
The Dockerfile provided also sets up CUDA GPU-accelerated neural network training in Tensorflow.
Note that the Dockerfile does not automatically install the python dependencies.

In order for the framework to function it requires a **grammar**, **evaluator** and **model**. This repository includes example grammars and evaluators but an additional step is required to create the models.

Sample model architectures are included in the `models/json/` directory. A utility script is included that converts these `.json` files into usable `.h5` files. 
**After installing the dependencies** use the following command to create the models:

```bash
> python -m utils.create_models
```

With the models created we can start using the framework. In order to run a sample experiment one can use the following command:

```bash
> python -m main
```

## How to use
In order to run your own experiments you will need to change the configuration used by the framework. While it is possible to set evey parameter manually through the command line (see `sge/parameters.py`) this approach is laborous and error prone. The recommended approach is to create your own parameter configuration file in the `parameters/` directory and loading it before the experiment. This can be done through the following command:

```bash
> python -m main --parameters parameters/your_configuration_file.yml
```

Additional guidelines on how to create custom evaluators coming soon.


