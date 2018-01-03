#! /usr/bin/env python3
"""
Method for loading and evaluating Tensorflow models.
"""
import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
import argparse
from spacy.en import English

from tensorflow.contrib import learn

import train

from tsa_inf_cnn import TsaInfCnnMapping
sys.path.append("/datadrive/ML/jasper/python/data_wrangler/")
import eval
import text_to_embeddings
import data_wrangler
sys.path.append('/datadrive/ML/jasper/python/embedding-evaluation/wordsim/')
from wordsim import Wordsim

FORMAT_ID = ["jinho", "godin"]


def config_eval_multi_test(config, verbose=0, as_ensemble=True):
    """
    Load and evaluate model/s based on a config file.

    :param config: dict {string:int/float/string}
        Dictionary of model/s and data parameters
    :param verbose: int {0, 1, 2}
        Set for more verbose output, passed on to method calls
    :param as_ensemble: boolean
        Set for evaluating multiple models (including cross validation models) as ensemble as opposed to by mean scores
    :return: float, float
        Evaluation score (mean if as_ensemble False, ensemble if as_ensemble True), standard deviation (only supported
        for as_ensemble False)
    """

    if config["run_name"] is not None:
        n_models = len(config["run_name"])
    else:
        raise ValueError("Implementation expects run_name parameter, but config['run_name'] is None.")

    ensemble_output_scores = []
    std = 0
    # Assume ensemble of models and run only first model if not
    for model_id in range(n_models):
        run_name_tmp = config["run_name"][model_id]

        run_format = None
        # Identify run_name format
        for format in FORMAT_ID:
            if format in run_name_tmp:
                run_format = format
                continue

        # get format test set
        for path in config["test_path"]:
            if run_format in path:
                test_path = path

        print("Identified {0:s} as test input for run {1:s}".format(test_path, run_name_tmp))

        # Identify test_path in case of multiple (embedding) test formats
        # copy config
        test_path_config = config.copy()
        test_path_config["test_path"] = test_path

        # Load numpy data
        start = time.time()
        print("Loading Data, this might take a while....")
        if test_path_config["input_type"] == "npy":
            x_test, y_test, _ = data_helper.load_data_npy(test_path_config["test_path"], test_path_config["n_classes"],
                                                          test_path_config["one_hot"])
        elif test_path_config["input_type"] == "npy_lex":
            x_test, _, y_test, _ = data_helper.load_data_lex_npy(
                test_path_config["test_path"],
                test_path_config["n_classes"],
                test_path_config["lexicon_embedding_dimension"]*test_path_config["document_length"]
            )
        elif test_path_config["input_type"] == "txt":
            _, y_test, x_test = text_to_embeddings.text_to_embeddings(
                test_path_config["test_path"],
                test_path_config["embedd_model_path"],
                test_path_config["embedding_dimension"],
                test_path_config["document_length"],
                one_hot=True
            )
        else:
            raise ValueError("Invalid input type: {0:s}".format(test_path_config["input_type"]))
        end = time.time()
        print("Loaded " + str(len(x_test)) + " dimension ( " + str(len(x_test[0])) + " ) train data items in " +
              str(end-start) + " seconds")

        print("\nEvaluating...\n")

        eval_score, std = model_inference(as_ensemble, test_path_config, ensemble_output_scores, model_id, std,
                                          verbose, x_test, y_test)
        # Calculate and print ensemble F1 if more than 1 model
        if n_models > 1:
            if as_ensemble:
                y_score = np.mean(np.asarray(ensemble_output_scores), axis=0)
                y_pred = np.argmax(y_score, 1)
                y_gold = np.argmax(y_test, 1)
                eval_score = eval.get_eval_metric(y_gold, y_pred, test_path_config["eval_metric"],
                                                  test_path_config["eval_labels"])
                print("The ensemble {0:s} score ({1:d}/{2:d}) is: {3:s}"
                      .format(test_path_config["eval_metric"], model_id+1, n_models, str(eval_score)))
            else:
                raise ValueError("Evaluating multiple models by mean and standard deviation (as opposed to as "
                                 "ensemble) is currently not supported.")
        else:
            y_score = ensemble_output_scores[0]
            y_pred = np.argmax(y_score, 1)
            y_gold = np.argmax(y_test, 1)
            eval_score = eval.get_eval_metric(y_gold, y_pred, test_path_config["eval_metric"],
                                              test_path_config["eval_labels"])
            print("The {0:s} score ({1:d}/{2:d}) is: {3:s}"
                  .format(test_path_config["eval_metric"], model_id + 1, n_models, str(eval_score)))

    return eval_score, std, y_pred, y_score


def config_eval(config, verbose=0, as_ensemble=True):
    """
    Load and evaluate model/s based on a config file.

    :param config: dict {string:int/float/string}
        Dictionary of model/s and data parameters
    :param verbose: int {0, 1, 2}
        Set for more verbose output, passed on to method calls
    :param as_ensemble: boolean
        Set for evaluating multiple models (including cross validation models) as ensemble as opposed to by mean scores
    :return: float, float
        Evaluation score (mean if as_ensemble False, ensemble if as_ensemble True), standard deviation (only supported
        for as_ensemble False)
    """
    # Load numpy data
    start = time.time()
    print("Loading Data, this might take a while....")
    if config["input_type"] == "npy":
        x_test, y_test, _ = data_helper.load_data_npy(config["test_path"], config["n_classes"], config["one_hot"])
    elif config["input_type"] == "npy_lex":
        x_test, _, y_test, _ = data_helper.load_data_lex_npy(
            config["test_path"],
            config["n_classes"],
            config["lexicon_embedding_dimension"]*config["document_length"]
        )
    elif config["input_type"] == "txt":
        _, y_test, x_test = text_to_embeddings.text_to_embeddings(
            config["test_path"],
            config["embedd_model_path"],
            config["embedding_dimension"],
            config["document_length"],
            one_hot=True
        )
    else:
        raise ValueError("Invalid input type: {0:s}".format(config["input_type"]))
    end = time.time()
    print("Loaded " + str(len(x_test)) + " dimension ( " + str(len(x_test[0])) + " ) train data items in " +
          str(end-start) + " seconds")

    print("\nEvaluating...\n")

    if config["checkpoint_file"] is not None:
        n_models = len(config["checkpoint_file"])
    else:
        n_models = len(config["run_name"])

    # Assume ensemble of models and run only first model if not
    ensemble_output_scores = []
    std = 0
    for model_id in range(n_models):
        eval_score, std = model_inference(as_ensemble, config, ensemble_output_scores, model_id, std, verbose, x_test,
                                          y_test)
        # Calculate and print ensemble F1 if more than 1 model
        if n_models > 1:
            if as_ensemble:
                y_score = np.mean(np.asarray(ensemble_output_scores), axis=0)
                y_pred = np.argmax(y_score, 1)
                y_gold = np.argmax(y_test, 1)
                eval_score = eval.get_eval_metric(y_gold, y_pred, config["eval_metric"], config["eval_labels"])
                print("The ensemble {0:s} score ({1:d}/{2:d}) is: {3:s}"
                      .format(config["eval_metric"], model_id+1, n_models, str(eval_score)))
            else:
                raise ValueError("Evaluating multiple models by mean and standard deviation (as opposed to as "
                                 "ensemble) is currently not supported.")
        else:
            y_score = ensemble_output_scores[0]
            y_pred = np.argmax(y_score, 1)
            y_gold = np.argmax(y_test, 1)
            eval_score = eval.get_eval_metric(y_gold, y_pred, config["eval_metric"], config["eval_labels"])
            print("The {0:s} score ({1:d}/{2:d}) is: {3:s}"
                  .format(config["eval_metric"], model_id + 1, n_models, str(eval_score)))

    return eval_score, std, y_pred, y_score


def model_inference(as_ensemble, config, ensemble_output_scores, model_id, std, verbose, x_test, y_test):
    if config["cross_valid"][model_id] != 0:
        eval_scores_cv = []
        cv_output_scores = []
        for cross_valid_id in range(config["cross_valid"][model_id]):
            architecture = config["architecture"][model_id]
            if config["checkpoint_file"] is not None:
                checkpoint_file = config["checkpoint_file"][model_id]
            else:
                run_name = train.cross_valid_run_name(config["run_name"][model_id], cross_valid_id)
                run_suffix = config["run_suffix"][model_id]
                checkpoint_file = train.get_model_path(run_name, run_suffix)

            variable_scope = config["variable_scope"][model_id]

            # Evaluation
            loss, eval_metric, conf_mat, output_scores = model_eval(
                architecture,
                checkpoint_file,
                x_test,
                y_test,
                eval_metric=config["eval_metric"],
                eval_labels=config["eval_labels"],
                variable_scope=variable_scope,
                verbose=verbose
            )
            eval_scores_cv.append(eval_metric)
            cv_output_scores.append(output_scores)
            if verbose >= 2:
                print(conf_mat)

        # Calculate and print cross valid ensemble F1
        cv_output_scores = np.asarray(cv_output_scores)
        mean_output_scores = np.mean(cv_output_scores, axis=0)
        ensemble_output_scores.append(mean_output_scores)
        if as_ensemble:
            y_pred = np.argmax(mean_output_scores, 1)
            y_gold = np.argmax(y_test, 1)
            eval_score = eval.get_eval_metric(y_gold, y_pred, config["eval_metric"], config["eval_labels"])
            print("The cross val ensemble {0:s} score is: {1:s}".format(config["eval_metric"], str(eval_score)))

        # Calculate and print mean F1
        eval_scores_cv = np.asarray(eval_scores_cv)
        eval_score = np.mean(eval_scores_cv)
        std = np.std(eval_scores_cv)
        print("The mean {0:s} score (cross valid models) is: {1:s}"
              .format(config["eval_metric"], str(eval_score)))
        # Add the print statement of printing the config
        print("The standard deviation (cross valid models) is: {0:s}".format(str(std)))

    else:
        architecture = config["architecture"][model_id]
        if config["checkpoint_file"] is not None:
            checkpoint_file = config["checkpoint_file"][model_id]
        else:
            run_name = train.cross_valid_run_name(config["run_name"][model_id], cross_valid_id)
            run_suffix = config["run_suffix"][model_id]
            checkpoint_file = train.get_model_path(run_name, run_suffix)
        variable_scope = config["variable_scope"][model_id]

        # Evaluation
        loss, eval_score, conf_mat, scores = model_eval(
            architecture,
            checkpoint_file,
            x_test,
            y_test,
            eval_metric=config["eval_metric"],
            eval_labels=config["eval_labels"],
            variable_scope=variable_scope,
            verbose=verbose
        )
        print("The {0:s} score is: {1:s}".format(config["eval_metric"], str(eval_score)))
        ensemble_output_scores.append(scores)
        if verboise >= 2:
            print(conf_mat)

    return eval_score, std


def model_eval(architecture, checkpoint_file, x_test, y_test, eval_metric="f1_macro", eval_labels=None,
               variable_scope="", verbose=0):
    """
    Load and evaluate a single model.

    :param architecture: string
        Unique architecture string
    :param checkpoint_file: string
        Path to Tensorflow checkpoint file
    :param x_test: numpy array
        Input data in the form of document embeddings
    :param y_test: numpy array
        Gold data in the form of one hot vectors
    :param eval_metric: string
        Identifier for eval metric {f1_macro, f1_micro}
    :param eval_labels: array
        Sett of labels to include. For multilabel targets, labels are column indices. By default (None), all labels in
        y_true and y_pred are used in sorted order.
    :param variable_scope:  string
        Additional variable_scope for model loading (default: '')
    :param verbose: int {0, 1, 2}
        Set for more verbose output, passed on to method calls
    :return: float, float, array, numpy array
        Loss, macro averaged F1, confustion matrix, model output scores
    """
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            if verbose >= 2:
                print("Loading checkpoint file: " + checkpoint_file)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            if verbose >= 2:
                print("Initializing model: " + architecture + "...")
            if architecture == "tsainfcnn":
                model = TsaInfCnnMapping(sess.graph, variable_scope=variable_scope)
            else:
                print("ERROR: unrecognized architecture " + architecture)
            if verbose >= 1:
                print("Model '{0:s}' initialized.".format(architecture))

            loss, eval_metric, conf_mat, scores = train.eval_step(
                model,
                sess,
                x_test,
                y_test,
                eval_metric=eval_metric,
                eval_labels=eval_labels,
                verbose=verbose
            )

    return loss, eval_metric, conf_mat, scores


def experiment_eval(config, verbose=0):
    """
    Find all experiments starting with run_name parameter and evaluate them individually.

    :param config: dict {string:int/float/string}
        Dictionary of model/s and data parameters
    :param verbose: int {0, 1, 2}
        Set for more verbose output, passed on to method calls
    :return: None
    """
    experiment_identifier = config["run_name"][0]

    # Find all models
    model_dir = train.MODEL_OUTPUT_PATH
    run_name_candidates = []
    for run_name in os.listdir(model_dir):
        if run_name.startswith(experiment_identifier):
            run_name_candidates.append(run_name)

    # Remove cv identifier
    run_names = []
    for run_candidate in run_name_candidates:
        run_identifier = run_candidate.replace(experiment_identifier, "")
        if "_cv" in run_identifier:
            run_identifier = run_identifier[0:run_identifier.index("_cv")]
            run_name = experiment_identifier + run_identifier
            if run_name not in run_names:
                run_names.append(run_name)

    if verbose >= 1:
        print("Found {0:d} runs of experiment {1:s}, starting evaluation:"
              .format(len(run_names), experiment_identifier))

    # Eval
    for run_name in run_names:
        config["run_name"] = [run_name]
        if verbose >= 1:
            print("Evaluating: {0:s}".format(run_name))
        config_eval(config, verbose=verbose, as_ensemble=False)

    return None


def smm4h_inference_old(config, verbose=0, as_ensemble=True):

    print("Loading embedding model, this can take some time...")
    model = Wordsim.load_vector(config["embedd_model_path"])
    print("Loading spaCy model, this can take some time...")
    nlp = English()

    if config["checkpoint_file"] is not None:
        n_models = len(config["checkpoint_file"])
    else:
        n_models = len(config["run_name"])

    # Assume ensemble of models and run only first model if not
    ensemble_output_scores = []
    std = 0

    # Read lines
    with open(config["submission_out"], "w", encoding='utf-8') as out_file:
        count = 0
        with open(config["test_path"], 'r', encoding='utf-8') as in_file:
            for line in in_file:
                # Get data
                rows = line.strip().split("\t")
                text = rows[1]

                # Convert to embedding
                doc_vector = text_to_embeddings.text_to_vec(
                    model,
                    nlp,
                    text,
                    config["embedding_dimension"],
                    config["document_length"]
                )
                x_test = doc_vector.reshape(1, config["embedding_dimension"] * config["document_length"])

                # Dummy label
                y_test = np.empty([1, config["n_classes"]])

                # Get prediction
                for model_id in range(n_models):
                    eval_score, std = model_inference(as_ensemble, config, ensemble_output_scores, model_id, std,
                                                      verbose, x_test, y_test)

                mean_ensemble_scores = np.mean(np.asarray(ensemble_output_scores), axis=0)
                y_pred = np.argmax(mean_ensemble_scores, 1)

                # Write
                out_file.write(line.strip() + "\t" + str(y_pred[0]) + "\n")
                count += 1
    return None


def smm4h_inference(config, verbose=0, as_ensemble=True):
    DEBUG = True
    # for each test file
    ensemble_output_scores = []
    n_runs = 0
    for test_path in config["test_path"]:
        test_format = None
        # check contains FORMAT_ID
        for format in FORMAT_ID:
            if format in test_path:
                test_format = format
                continue

        # get FORMAT_ID models
        run_name_tmp = []
        for run in config["run_name"]:
            if test_format in run:
                run_name_tmp.append(run)
        n_runs_tmp = len(run_name_tmp)

        print("Identified {0:d} runs for test input {1:s}:\n{2:s}".format(n_runs_tmp, test_path, str(run_name_tmp)))

        # copy config
        test_path_config = config.copy()
        test_path_config["test_path"] = test_path
        test_path_config["run_name"] = run_name_tmp

        # run
        _, _, _, y_score_tmp = config_eval(test_path_config, verbose, as_ensemble)

        # Append prediction X n_models
        weighted_scores = y_score_tmp * n_runs_tmp
        ensemble_output_scores.append(weighted_scores)
        if DEBUG:
            print("{0:s} y_score_tmp[0:10, :]: {1:s}".format(test_format, str(y_score_tmp[0:10, :])))
            print("{0:s} weighted_scores[0:10, :]: {1:s}".format(test_format, str(weighted_scores[0:10, :])))

        n_runs += n_runs_tmp

    # Average prediction
    mean_ensemble_scores = np.sum(np.asarray(ensemble_output_scores), axis=0) / n_runs
    if DEBUG:
        print("mean_ensemble_scores[0:10, :]: {0:s}".format(str(mean_ensemble_scores[0:10, :])))
    y_pred = np.argmax(mean_ensemble_scores, 1)

    # Read lines
    with open(config["submission_out"], "w", encoding='utf-8') as out_file:
        count = 0
        with open(config["submission_in"], 'r', encoding='utf-8') as in_file:
            for line in in_file:
                # Write
                out_file.write(line.strip() + "\t" + str(y_pred[count] + 1) + "\n")
                count += 1

    if verbose > 1:
        print("Wrote {0:d}/{1:d} labels to: {2:s}".format(count, y_pred.shape[1], config["submission_out"]))
    return None


def main():
    # Parameters
    parser = argparse.ArgumentParser()
    # Experiment parameters
    parser.add_argument('--verbose', '-v', type=int, default=0,
                        help="Set for more verbose output {0 - off, 1 - some, 2 -all} (default: 0).")
    parser.add_argument('--eval_metric', default="f1_macro",
                        help="Metric for saving best model and evaluating early stop "
                             "{loss, f1_macro, f1_micro, recall_macro} (default: f1_macro)")
    parser.add_argument('--eval_labels', type=int, nargs='+', default=None,
                        help="Labels for eval metric, None means all (default: None)")
    parser.add_argument('--experiment_eval', action='store_true',
                        help="Set to evaluate all experiments defined by --run_name (default: False)")
    parser.add_argument('--submission_out', help="Output path for test file with prediction annotation.")
    parser.add_argument('--submission_in',
                        help="Input path for submission file, system will append '\t' followed by test_path label.")
    # Data parameters
    parser.add_argument('--test_path', '-tst', nargs='+', help="Path to evaluation data")
    parser.add_argument('--input_type', default="npy",
                        help="Type of input data {'npy', 'txt', 'npy_lex'} (default: 'npy')")
    parser.add_argument('--embedd_model_path', help="Path to embedding model, only required if input_type='txt'")
    parser.add_argument('--no_one_hot', dest='one_hot', action='store_false')
    parser.add_argument('--document_length', '-d', type=int, default=120,
                        help="Number of input tokens per document (default: 120)")
    parser.add_argument('--embedding_dimension', '-e', type=int, required=True,
                        help="Dimension of input word embeddings")
    parser.add_argument('--n_classes', '-c', type=int, required=True, help="Number of classes in data")
    parser.add_argument('--lexicon_embedding_dimension', type=int, default=0,
                        help="Dimension of input lexicon dimensions (default: 0, ignore)")
    # Model loading parameters
    parser.add_argument('--architecture', '-a', nargs='+', required=True, default=["tsainfcnn"],
                        help="select the deep learning architecture, options = {tsainfcnn} (default: [tsainfcnn])")
    parser.add_argument('--run_name', '-r', nargs='+',
                        help="Name of training run, determines the model name and checkpoint location.")
    parser.add_argument('--run_suffix', default=["best_model"], nargs='+',
                        help="Suffix for identifying checkpoint of multiple saved models (default: ['best_model'] ")
    parser.add_argument('--checkpoint_file', nargs='+',
                        help="Path to checkpoint file from training run, alternative to run_name")
    parser.add_argument('--variable_scope', type=str, default=[""], nargs='+',
                        help="Additional variable scope for loading model parameters (default: [''])")
    parser.add_argument('--cross_valid', nargs='+', type=int, default=[0],
                        help="Value of k for cross-validation (default: [0]")

    args = parser.parse_args()

    config = {}
    for arg in vars(args):
        print(arg + ': ' + str(getattr(args, arg)))
        config[arg] = getattr(args, arg)

    if config["input_type"] == "txt" and config["embedd_model_path"] is None:
        raise ValueError("Input type 'txt' requires embedd_model_path.")

    if config["experiment_eval"]:
        if config["run_name"] is None:
            raise ValueError("Parameter run_name required for experiment_eval.")

        experiment_eval(config, args.verbose)
    else:
        if config["run_name"] is None and config["checkpoint_file"] is None:
            raise ValueError("Either run_name or checkpoint_file input is required.")
        if config["run_name"] is not None and config["checkpoint_file"] is not None:
            raise ValueError("Parameters run_name or checkpoint_file are mutually exclusive.")

        if config["checkpoint_file"] is not None:
            n_models = len(config["checkpoint_file"])
        else:
            n_models = len(config["run_name"])

        if n_models > 1:
            # Check default inputs for ensemble
            fix_parameter_length(config, n_models, "architecture")
            fix_parameter_length(config, n_models, "run_suffix")
            fix_parameter_length(config, n_models, "variable_scope")
            fix_parameter_length(config, n_models, "cross_valid")

        if config["submission_out"] is not None:
            smm4h_inference(config, args.verbose)
        else:
            if len(config["test_path"]) > 1:
                print("Found multiple test_path inputs. Assuming different formats of the same data. Will match to runs"
                      "by format identifiers: {0:s}".format(str(FORMAT_ID)))
                config_eval_multi_test(config, args.verbose)
            else:
                config["test_path"] = config["test_path"][0]
                config_eval(config, args.verbose)

    return None


def fix_parameter_length(config, expected_length, parameter):
    """
    Expands parameter list value to expected length.

    No changes if parameter list already has expected length. Only expands from parameter length 1 and raises
    ValueError otherwise.

    :param config:
    :param expected_length:
    :param parameter:
    :return:
    """
    if len(config[parameter]) != expected_length:
        if len(config[parameter]) != 1:
            raise ValueError("Invalid input length {0:d} for {1:s}. Must be length {2:d} as determined by"
                             "--run_name/--checkpoint_file or length 1 (default, same for all)."
                             .format(len(config[parameter]), parameter, expected_length))

        config[parameter] = config[parameter] * expected_length
    return None

if __name__ == "__main__":
    main()
