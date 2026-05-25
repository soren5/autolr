import argparse
import os
import yaml

""""Algorithm Parameters"""
default_params = {
    'PARAMETERS': None,
    'POPSIZE': 2,
    'GENERATIONS': 3,
    'ELITISM': 0,
    'PROB_CROSSOVER': 0.9,
    'PROB_MUTATION': 0.15,
    'SELECTION_TYPE': 'tournament',
    'TSIZE': 2,
    'GRAMMAR': 'grammars/basic_optimizer.txt',
    'EXPERIMENT_NAME': "example",
    'RUN': 1,
    'INCLUDE_GENOTYPE': True,
    'SAVE_STEP': 1,
    'VERBOSE': True,
    'MIN_TREE_DEPTH': 6,
    'MAX_TREE_DEPTH': 17,
    'MODEL': 'mnist_model.h5',
    'DATASET': 'fmnist',
    'TRAINING_SIZE': 3500,
    'VALIDATION_SIZE': 3500,
    'FITNESS_SIZE': 53000,
    'BATCH_SIZE': 5,
    'EPOCHS': 5,
    'SEED': None,
    'PREPOPULATE': False,
    'FITNESS_FLOOR': 0,
    'LOAD_ARCHIVE': True,
    'CURRENT_GEN': -1,
    'SINGLE_GEN': False,
    'MULTI_TASK': False,
    'FAKE_FITNESS': False,
    'NORMALIZE': True,
    'SUBTRACT_MEAN': True,
    
    'DUMPS_DIR': os.environ.get('DUMPS_DIR', 'dumps'),
    'LOGS_DIR': os.environ.get('LOGS_DIR', 'logs'),
    'DATA_DIR': os.environ.get('DATA_DIR', 'data'),
    'MODELS_DIR': os.environ.get('MODELS_DIR', 'models'),
    
    # Early stop settings, only used if patience is a positive integer
    'PATIENCE': False,
    'VALIDATION_METRIC': 'val_accuracy',
    'MIN_DELTA': 0.0,
    
    # Multi task thresholds, only used when MULTI_TASK is True
    'FMNIST_THRESHOLD': 0.8,
    'CIFAR10_THRESHOLD': 0.7,
    'CIFAR100_THRESHOLD': 0.5,
    'TINY_IMAGENET_THRESHOLD': 0.3,
    
    # F-race settings, only used when RACING is True
    'RACING': False,
    'RACING_ALPHA': 0.05,
    'RACING_MIN_EVALS': 2,
    'RACING_MAX_EVALS': 30,
    'RACING_STAT_TEST': 'mannwhitney',
    'RACING_DECISION_RULE': 'scalar_mannwhitney',
    'RACING_LOGGING': True,
    'RACING_SELECTION_AUDIT': True,
    }

params = default_params.copy()

def load_parameters(file_name="parameters/adaptive_autolr.yml"):
    with open(file_name, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print("using ",file_name, "for parameters")
    params.update(cfg)

def reset_parameters():
    params.clear()
    params.update(default_params.copy())

def manual_load_parameters(parameters):
    reset_parameters()
    params.update(parameters)
    
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
    parser.add_argument('--single_gen',
        dest="SINGLE_GEN",
        type=bool,
        help="If true, only one generation is run, and the program ends.")
    parser.add_argument('--racing',
        dest="RACING",
        type=bool,
        help="Enables within-generation F-race reevaluation.")
    parser.add_argument('--racing_alpha',
        dest="RACING_ALPHA",
        type=float,
        help="Significance threshold used by F-race.")
    parser.add_argument('--racing_min_evals',
        dest="RACING_MIN_EVALS",
        type=int,
        help="Minimum evaluations before F-race can eliminate a candidate.")
    parser.add_argument('--racing_max_evals',
        dest="RACING_MAX_EVALS",
        type=int,
        help="Maximum total evaluations per archive key during F-race.")
    parser.add_argument('--racing_stat_test',
        dest="RACING_STAT_TEST",
        type=str,
        help="Statistical test used by F-race. Currently only 'mannwhitney' is supported.")
    parser.add_argument('--racing_decision_rule',
        dest="RACING_DECISION_RULE",
        type=str,
        help="F-race decision rule. Supported values: scalar_mannwhitney, gated_cascade.")
    parser.add_argument('--racing_logging',
        dest="RACING_LOGGING",
        type=bool,
        help="Enables F-race structured logging when racing is enabled.")
    parser.add_argument('--racing_selection_audit',
        dest="RACING_SELECTION_AUDIT",
        type=bool,
        help="Enables F-race parent selection and elitism counterfactual audit.")
    

    
    
        
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
