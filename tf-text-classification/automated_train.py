import train
from random import shuffle
import time
import argparse
import numpy as np
import copy
import sys
import os

DEBUG = False


def permute_configs(configs, parameter_name, parameter_range):
    """
    Permute each config with all elements of parameter range.

    :param configs: list of dict {string:int/float/string}
        List of stand alone parameter combination defining training experiments
    :param parameter_name: string
        Name of parameter to append to input configs
    :param parameter_range: list of int/float/string
        Parameter range over which to permute each input config
    :return: list of dict {string:int/float/string}
        Permutation of each input config with each element of parameter range
    """
    output_configs = []
    for config in configs:
        for parameter in parameter_range:
            new_config = copy.copy(config)
            new_config[parameter_name] = parameter
            output_configs.append(new_config)
    return output_configs


def permutation_param_tuning(architecture, batch_size_range, cross_valid, dev_path, document_length,
                             embedding_dimension, experiment_name, keep_prob_range,
                             learning_rate_range, lexicon_embedding_dimension, n_classes,
                             num_epochs, one_hot, save_iteration,
                             test_path, train_path, early_stop_patience, architecture_config):
    """
    Create permutations over input parameter ranges.

    :param architecture:
    :param batch_size_range:
    :param cross_valid:
    :param dev_path:
    :param document_length:
    :param embedding_dimension:
    :param experiment_name:
    :param keep_prob_range:
    :param learning_rate_range:
    :param lexicon_embedding_dimension:
    :param n_classes:
    :param num_epochs:
    :param one_hot:
    :param save_iteration:
    :param test_path:
    :param train_path:
    :param early_stop_patience:
    :param architecture_config:
    :return: configs
    """
    # (TODO): Separate into training parameter config and hyperparameter config
    # TODO: Generalize to work on any config independent of parameter names

    configs = []
    for batch_size in batch_size_range:
        config = {"batch_size": batch_size}
        configs.append(config)

    configs = permute_configs(configs, "embedd_filter_sizes", architecture_config["embedd_filter_sizes_range"])

    configs = permute_configs(configs, "keep_prob", keep_prob_range)

    configs = permute_configs(configs, "n_dense_output", architecture_config["n_dense_output_range"])

    configs = permute_configs(configs, "n_filters", architecture_config["n_filters_range"])

    configs = permute_configs(configs, "learning_rate", learning_rate_range)

    configs = permute_configs(configs, "annealing_restarts", architecture_config["annealing_restarts_range"])

    configs = permute_configs(configs, "annealing_factor", architecture_config["annealing_factor_range"])

    if architecture_config["architecture"] in "attention-cnn":
        configs = permute_configs(configs, "attention_depth", architecture_config["attention_depth_range"])

    if architecture_config["optimizer"] == "adam":
        configs = permute_configs(configs, "adam_b2", architecture_config["adam_b2_range"])

    final_configs = []
    for i in range(len(configs)):
        new_m = copy.copy(configs[i])
        # Number runs
        new_m["run_name"] = experiment_name + "_" + str(i)
        # Add non range parameters
        new_m["lexicon_embedding_dimension"] = lexicon_embedding_dimension
        new_m["test_path"] = test_path
        new_m["save_iteration"] = save_iteration
        new_m["num_epochs"] = num_epochs
        new_m["document_length"] = document_length
        new_m["embedding_dimension"] = embedding_dimension
        new_m["cross_valid"] = cross_valid
        new_m["architecture"] = architecture
        new_m["n_classes"] = n_classes
        new_m["train_path"] = train_path
        new_m["dev_path"] = dev_path
        new_m["one_hot"] = one_hot
        new_m["early_stop_patience"] = early_stop_patience
        new_m["optimizer"] = architecture_config["optimizer"]
        new_m["eval_metric"] = architecture_config["eval_metric"]
        new_m["eval_labels"] = architecture_config["eval_labels"]
        final_configs.append(new_m)
        #  print("Length of final_config list is :" + str(len(final_config)))

    return final_configs


def random_search(configs, max_time, append=True, shuffle_configs=True, verbose=0):
    """
    Random search over input configurations, limited by maximal time.

    :param configs: list of dict {string:int/float/string}
        List of stand alone parameter combination defining training experiments
    :param max_time: int
        Maximal time in hours after which to terminate (last experiment will finish)
    :param append: boolean
        Will check for and skip previous experiments instead of overwriting if True (default: True)
    :param shuffle_configs: boolean
        Run random set of configs instead of in order. Highly discouraged to set to False unless all experiments can be
        executed, which makes this grid search. (default: True)
    :param verbose: int {0, 1, 2}
        Set for more verbose output, passed on to method calls
    :return: None
    """
    n_configs = len(configs)

    if verbose >= 1:
        print("Will run experiments out of {0:d} config permutations until max time of {1:d} hours is exceeded"
              .format(n_configs, max_time))

    if shuffle_configs is True:
        print("Shuffling config permutations.")
        shuffle(configs)

    total_time = 0
    count = 0
    for elem in configs:
        if append:
            if experiment_output_exists(elem):
                print("Output for {0:s} detected, will skip experiment".format(elem['run_name']))
                continue
        start_time = time.time()
        #  print("Start time is:" + str(start_time))
        print("")
        print("Experiment {0:d}/{1:d}:".format(count, n_configs))
        for key, val in elem.items():
            print(str(key) + ": " + str(val))
        train.full_train(elem, verbose)
        count += 1
        end_time = time.time()
        experiment_time = end_time - start_time
        #  print("Experiment took: " + str(experiment_time) + " seconds")
        total_time = total_time + experiment_time
        total_time_in_hrs = total_time / 3600
        if DEBUG:
            print("DEBUG: Only running one config for debugging. Finished.")
            break
        if total_time_in_hrs > max_time:
            break

    if verbose >= 1:
        print("Ran {0:d}/{1:d} experiments in {2:.2f} hours".format(count, n_configs, total_time_in_hrs))

    if shuffle_configs is False and count < n_configs:
        print("WARN: Only ran {0:d}/{1:d} experiments, without shuffling. Shuffling or running all experiments is "
              "advised for relevant parameter search results.".format(count, n_configs))

    return None


def experiment_output_exists(config):
    """
    Check if output files for experiment, defined by run_name config parameter, already exist.

    This does not guarantee that the experiment was run successfully and completely, just that it was run at some point.

    :param config: dict {string:int/float/string}
        Stand alone parameter combination defining training experiments
    :return: boolean
        True if experiment output found, else False
    """
    path = os.path.abspath(os.path.curdir)
    path = os.path.join(path, train.MODEL_OUTPUT_PATH)
    path = os.path.join(path, config['run_name'])

    return os.path.exists(path)


def main():
    """
    Run random parameter search based on argparse arguments.

    For example:
    python automated_train.py \
    --max_running_time 20 \
    --cross_valid 5 \
    --experiment_name smm4h_task2_godin_cv_shub \
    --architecture tsainfcnn \
    --learning_rate_range 0.0001 0.001 \
    --num_epochs 50 \
    --batch_size_range 50 100 150 \
    --document_length  \
    --embedding_dimension 400 \
    --n_classes 3 \
    --train_path /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_train_godin_400.npy \
    --keep_prob_range 0.7 0.8 0.9 \
    --embedd_filter_sizes_range 2,3,4,5,6 1,2,3,4,5 \
    --n_filters_range 100 150 200 \
    --n_dense_output_range 100 150 200

    See automated_train_cmd.txt in same directory for more examples.

    :return: None
    """
    # List for high level param, that don't go into config objects
    high_level_param = []
    train_parser = argparse.ArgumentParser(add_help=False)
    # Hyper-parameter optimization parameters
    train_parser.add_argument('--max_running_time', type=int, default=12,
                              help="Maximum training time in hours, script will terminate when exceeded (default: 12)")
    train_parser.add_argument('--no_shuffle_configs', dest='shuffle_configs', action='store_false',
                              help="Set to turn off random shuffle of parameter permutations, highly discouraged unless"
                                   " all permutations are run")
    train_parser.add_argument('--experiment_name', '-r', required=True,
                              help="Name of self-service training run, determines the model names and checkpoint "
                                   "locations")
    train_parser.add_argument('--eval_metric', default="f1_macro",
                              help="Metric for saving best model and evaluating early stop "
                                   "{loss, f1_macro, f1_micro, recall_macro} (default: f1_macro)")
    train_parser.add_argument('--eval_labels', type=int, nargs='+', default=None,
                              help="Labels for eval metric, None means all (default: None)")
    train_parser.add_argument('--verbose', '-v', type=int, default=0,
                              help="Set for more verbose output {0 - off, 1 - some, 2 -all} (default: 0).")
    high_level_param.append("verbose")
    # Data parameters
    train_parser.add_argument('--train_path', '-trn', required=True, help="Path to training data")
    train_parser.add_argument('--dev_path', '-dev', help="Path to development/validation data")
    train_parser.add_argument('--test_path', '-tst', help="Path to testing data")
    train_parser.add_argument('--document_length', '-d', type=int, default=120,
                              help="Number of input tokens per document (default: 120)")
    train_parser.add_argument('--embedding_dimension', '-e', type=int, required=True,
                              help="Dimension of input word embeddings")
    train_parser.add_argument('--lexicon_embedding_dimension', type=int, default=0,
                              help="Dimension of input lexicon dimensions (default: 0, ignore)")
    train_parser.add_argument('--n_classes', '-c', type=int, required=True,
                              help="Number of classes in data, determines output dimension")
    train_parser.add_argument('--no_one_hot', dest='one_hot', action='store_false',
                              help="Set if labels in input data are not one-hot encoded")
    # Training parameters
    train_parser.add_argument('--cross_valid', type=int, default=0,
                              help="value of k for cross-validation (default: 0, train/dev split)")
    train_parser.add_argument('--save_iteration', type=int, default=0,
                              help="Aside from best model save each save_iterations-th model (default 0, off)")
    train_parser.add_argument('--early_stop_patience', type=int, default=3,
                              help="Number of epochs without improvement after which to stop training (default: 3)")
    # Hyper-parameters
    train_parser.add_argument('--optimizer', default="adam",
                              help="Optimizer for backpropagation {adam} (default: adam)")
    train_parser.add_argument('--adam_b2_range', nargs='+', type=float, default=[0.9, 0.999],
                              help="Adam decay rate for l2 norm moving average (default: '0.9 0.999')")
    train_parser.add_argument('--annealing_restarts_range', nargs='+', type=int, default=[2],
                              help="Range of learning rate annealing restarts {0 - off} (default: [2])")
    train_parser.add_argument('--annealing_factor_range', nargs='+', type=int, default=[2],
                              help="Division factor range for annealing learning rate (default: [2])")
    train_parser.add_argument('--architecture', '-a', required=True, default="tsainfcnn",
                              help="select the deep learning architecture, options = {tsainfcnn} (default: tsainfcnn)")
    train_parser.add_argument('--learning_rate_range', '-l', nargs='+', type=float, default=[0.0001, 0.001],
                              help="Range for learning rate rage (default: '0.0001 0.001')")
    train_parser.add_argument('--num_epochs', '-n', type=int, default=50,
                              help="Rnage for number of epochs over full training data (default: 50)")
    train_parser.add_argument('--batch_size_range', '-b', nargs='+', type=int, default=[50, 100, 150],
                              help="Size of training and eval batches (default: '50 100 150')")
    train_parser.add_argument('--keep_prob_range', nargs='+', type=float, default=[0.7, 0.8, 0.9],
                              help="Keep probability for dropout (default: '0.7 0.8 0.9')")
    # Architecture specific Hyper-parameters
    if "tsainfcnn" in sys.argv:
        tsainfcnn_parser = argparse.ArgumentParser(parents=[train_parser])
        tsainfcnn_parser.add_argument('--embedd_filter_sizes_range', nargs='+', type=str,
                                      default=["2,3,4,5,6", "1,2,3,4,5", "2,3,3,3,5"],
                                      help="Range for filter dimensions of word embedding CNN convolutions "
                                           "(default: '2,3,4,5,6 1,2,3,4,5 2,3,3,3,5')")
        tsainfcnn_parser.add_argument('--n_filters_range', nargs='+', type=int, default=[100, 150],
                                      help="Range for number of convolutions per filter size (default: '100 150')")
        tsainfcnn_parser.add_argument('--n_dense_output_range', nargs='+', type=int, default=[100, 150, 200, 250],
                                      help="Range for number of units in dense layer (default: '100 150 200 250')")
        args = tsainfcnn_parser.parse_args()
        args.embedd_filter_sizes_range = [list(map(int, filter_sizes_str.split(","))) for filter_sizes_str in
                                          args.embedd_filter_sizes_range]
    elif "attention-cnn" in sys.argv:
        attention_cnn_parser = argparse.ArgumentParser(parents=[train_parser])
        attention_cnn_parser.add_argument('--embedd_filter_sizes_range', nargs='+', type=str,
                                          default=["2,3,4,5,6", "1,2,3,4,5", "2,3,3,3,5"],
                                          help="Range for filter dimensions of word embedding CNN convolutions "
                                               "(default: '2,3,4,5,6 1,2,3,4,5 2,3,3,3,5')")
        attention_cnn_parser.add_argument('--n_filters_range', nargs='+', type=int, default=[100, 150],
                                          help="Range for number of convolutions per filter size (default: '100 150')")
        attention_cnn_parser.add_argument('--n_dense_output_range', nargs='+', type=int, default=[100, 150, 200, 250],
                                          help="Range for number of units in dense layer (default: '100 150 200 250')")
        attention_cnn_parser.add_argument('--attention_depth_range', nargs='+', type=int, default=[50],
                                          help="Depth of attention vector (default: 50)")
        args = attention_cnn_parser.parse_args()
        args.embedd_filter_sizes_range = [list(map(int, filter_sizes_str.split(","))) for filter_sizes_str in
                                          args.embedd_filter_sizes_range]
    else:
        raise ValueError("Could not identifier architecture argument.")

    config = {}
    print("ARGS:")
    for arg in vars(args):
        print(arg + ': ' + str(getattr(args, arg)))
        if arg not in high_level_param:
            config[arg] = getattr(args, arg)
    print("")

    if DEBUG:
        print("DEBUG: Setting num_epochs=3 for debugging.")
        args.num_epochs = 3

    # Create permutations over input parameter ranges
    configs = permutation_param_tuning(args.architecture, args.batch_size_range, args.cross_valid, args.dev_path,
                                       args.document_length, args.embedding_dimension,
                                       args.experiment_name, args.keep_prob_range, args.learning_rate_range,
                                       args.lexicon_embedding_dimension, args.n_classes,
                                       args.num_epochs, args.one_hot,
                                       args.save_iteration, args.test_path, args.train_path, args.early_stop_patience,
                                       config)
    # Run random subset of configs, checking for and skipping previously run experiments (append=True)
    random_search(configs, args.max_running_time, append=True, shuffle_configs=args.shuffle_configs, verbose=args.verbose)

    return None


if __name__ == "__main__":
    main()
