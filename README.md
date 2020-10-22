# AutoLR: An evolutionary framework for learning rate optimizers
AutoLR is a framework based on the [SGE](http://https://github.com/nunolourenco/sge3 "SGE") engine capable of evolving learning rate optimizers for specific neural network architectures and problems. This repository is part of a published work; if you end up using this framework it would be appreciated that you reference the following:
```
@inproceedings{carvalho20,
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


This system utilizes a grammar to generate a population of potential optimizers. The quality of these optimizers is then assessed using an evaluator. The best optimizers are then combined amongst themselves and altered in order to create a new population.

## Quickstart
This section will briefly explain how to run a simple AutoLR experiment.
The repository includes a `requirements.txt` file that should be used to install all necessary dependencies using the command:

```bash
> pip install -r requirements.txt
```

In order for the framework to function it requires a **grammar**, **evaluator** and **model**. This repository includes example grammars and evaluators but an additional step is required to create the models.

Sample model architectures are included in the `models/json/` directory. A utility script is included that converts these `.json` files into usable `.h5` files. 
**After installing the dependencies** use the following command to create the models:

```bash
> python -m utils.create_models
```

With the models created we can start using the framework. In order to run a sample experiment one can use the following command:

```bash
> python -m examples.evolve_dynamic_optimizer
```

## How to use
In order to run your own experiments you will need to change the configuration used by the framework. While it is possible to set evey parameter manually through the command line (see `sge/parameters.py`) this approach is laborous and error prone. The recommended approach is to create your own parameter configuration file in the `parameters/` directory and loading it before the experiment. This can be done through the following command:

```bash
> python -m examples.evolve_dynamic_optimizer --parameters parameters/your_configuration_file.yml
```

Additional guidelines on how to create custom evaluators coming soon.


