import argparse

import yaml

""""Algorithm Parameters"""
params = {
    'PARAMETERS': None,
    'POPSIZE': 2,
    'GENERATIONS': 3,
    'ELITISM': 0,
    'PROB_CROSSOVER': 0.9,
    'PROB_MUTATION': 0.15,
    'SELECTION_TYPE': 'tournament',
    'TSIZE': 2,
    'GRAMMAR': 'grammars/adaptive_autolr_grammar.txt',
    'EXPERIMENT_NAME': "dumps/example",
    'RUN': 1,
    'INCLUDE_GENOTYPE': True,
    'SAVE_STEP': 1,
    'VERBOSE': True,
    'MIN_TREE_DEPTH': 6,
    'MAX_TREE_DEPTH': 17,
    'MODEL': 'models/mnist_model.h5',
    'DATASET': 'fmnist',
    'VALIDATION_SIZE': 100,
    'FITNESS_SIZE': 59890,
    'BATCH_SIZE': 5,
    'EPOCHS': 5,
    'SEED': None,
    'PREPOPULATE': False,
    'PATIENCE': False,
    'FITNESS_FLOOR': 0,
    'LOAD_ARCHIVE': True,
    }


def load_parameters(file_name="parameters/adaptive_autolr.yml"):
    with open(file_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print("using ",file_name, "for parameters")
    params.update(cfg)


def set_parameters(arguments):
    # Initialise parser
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        description="Welcome to AutoLR",
    )
    parser.add_argument('--parameters',
                        dest='PARAMETERS',
                        type=str,
                        help='Specifies the parameters file to be used. Must '
                             'include the full file extension. Full file path'
                             'does NOT need to be specified.')
    parser.add_argument('--popsize',
                        dest='POPSIZE',
                        type=int,
                        help='Specifies the population size.')
    parser.add_argument('--generations',
                        dest='GENERATIONS',
                        type=float,
                        help='Specifies the total number of generations.')
    parser.add_argument('--elitism',
                        dest='ELITISM',
                        type=int,
                        help='Specifies the total number of individuals that should survive in each generation.')
    parser.add_argument('--seed',
                        dest='SEED',
                        type=int,
                        help='Specifies the seed to be used by the random number generator.')
    parser.add_argument('--prob_crossover',
                        dest='PROB_CROSSOVER',
                        type=float,
                        help='Specifies the probability of crossover usage. Float required')
    parser.add_argument('--prob_mutation',
                        dest='PROB_MUTATION',
                        type=float,
                        help='Specifies the probability of mutation usage. Float required')
    parser.add_argument('--selection',
                        dest='SELECTION_TYPE',
                        type=int,
                        help='Specifies the type of selection, either tournament selection or stochastic sampling.')
    parser.add_argument('--tsize',
                        dest='TSIZE',
                        type=int,
                        help='Specifies the tournament size for parent selection.')
    parser.add_argument('--model',
                        dest='MODEL',
                        type=str,
                        help='Specifies the path to the model file.')
    parser.add_argument('--dataset',
                        dest='DATASET',
                        type=str,
                        help='Specifies the dataset to load.')
    parser.add_argument('--grammar',
                        dest='GRAMMAR',
                        type=str,
                        help='Specifies the path to the grammar file.')
    parser.add_argument('--experiment_name',
                        dest='EXPERIMENT_NAME',
                        type=str,
                        help='Specifies the name of the folder where stats are going to be stored')
    parser.add_argument('--run',
                        dest='RUN',
                        type=int,
                        help='Specifies the run number.')
    parser.add_argument('--include_genotype',
                        dest='INCLUDE_GENOTYPE',
                        type=bool,
                        help='Specifies if the genotype is to be include in the log files.')
    parser.add_argument('--save_step',
                        dest='SAVE_STEP',
                        type=int,
                        help='Specifies how often stats are saved')
    parser.add_argument('--verbose',
                        dest='VERBOSE',
                        type=bool,
                        help='Turns on the verbose output of the program')
    parser.add_argument('--resume',
                        dest="RESUME",
                        type=str,
                        help="")
    parser.add_argument('--prepopulate',
                        dest="PREPOPULATE",
                        type=bool,
                        help="")
    parser.add_argument('--protect',
                    dest="PROTECT",
                    type=bool,
                    help="")
    parser.add_argument('--genes',
                    dest="GENES",
                    type=str,
                    help="")
    parser.add_argument('--patience',
                dest="PATIENCE",
                type=int,
                help="")
    parser.add_argument('--validation_size',
                dest="VALIDATION_SIZE",
                type=int,
                help="")
    parser.add_argument('--test_size',
                dest="FITNESS_SIZE",
                type=int,
                help="")
    parser.add_argument('--batch_size',
                dest="BATCH_SIZE",
                type=int,
                help="")
    parser.add_argument('--epochs',
                dest="EPOCHS",
                type=int,
                help="")    
    parser.add_argument('--fake',
                dest="FAKE_FITNESS",
                type=bool,
                help="")
    parser.add_argument('--fitness_floor',
            dest="FITNESS_FLOOR",
            type=float,
            help="")
    parser.add_argument('--load_archive',
            dest="LOAD_ARCHIVE",
            type=bool,
            help="")
    parser.add_argument('--parent_experiment',
            dest="PARENT_EXPERIMENT",
            type=str,
            help="specifies in whihc folder to look for the parent run population and state (same run number), to use seed the current run (which must not have started already)")

    
    
        
    # Parse command line arguments using all above information.
    args, _ = parser.parse_known_args(arguments)

    # All default args in the parser are set to "None". Only take arguments
    # which are not "None", i.e. arguments which have been passed in from
    # the command line.
    cmd_args = {key: value for key, value in vars(args).items() if value is
                not None}

    # Set "None" values correctly.
    for key in sorted(cmd_args.keys()):
        # Check all specified arguments.

        if type(cmd_args[key]) == str and cmd_args[key].lower() == "none":
            # Allow for people not using correct capitalisation.
            cmd_args[key] = None

    if 'PARAMETERS' in cmd_args:
        load_parameters(cmd_args['PARAMETERS'])
    else:
       print("No parameter file found, using default parameters")


    params.update(cmd_args)

