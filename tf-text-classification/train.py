"""
Created on Sep 26, 2016

@author: Jasper_Friedrichs, Amit_Deshmane

Training script for various deep learning architectures
"""
import tensorflow as tf
import numpy
import os
import time
import data_helper
import sys
import time
import datetime
import math
sys.path.append('/datadrive/ML/jasper/python/data_wrangler/')
import eval
import argparse

from tsa_inf_cnn import TSAINFCNN, TsaInfCnnMapping
from attention_cnn import AttentionCNN, AttentionCNNMapping

MODEL_OUTPUT_PATH = "tmp"
DEBUG = False


def train(config, x_train, y_train, x_dev, y_dev, k_fold_id=None, verbose=0):
    """
    Initialize model defined by config and train.

    :param config:
    :param x_train:
    :param y_train:
    :param x_dev:
    :param y_dev:
    :param verbose:
    :return: conf_mat, int, object or list thereof
        Returns best confusion matrix, best epoch, best model or in case of annealing list thereof
    """
    #run in own graph so method can be called multiple times
    with tf.Graph().as_default():
        if verbose >= 2:
            print("Initializing model: " + config['architecture'] + "...")
        if config['architecture'] == "tsainfcnn":
            model = TSAINFCNN(config['document_length'], config['embedding_dimension'], config['n_classes'],
                              config['embedd_filter_sizes'], config['n_filters'], config['n_dense_output'])
        elif config['architecture'] == "attention-cnn":
            model = AttentionCNN(config['document_length'], config['embedding_dimension'], config['n_classes'],
                                 config['embedd_filter_sizes'], config['n_filters'], config['n_dense_output'],
                                 config['attention_depth'])
        else:
            print("ERROR: unrecognized architecture " + config['architecture'])
        if verbose >= 1:
            print("Model '{0:s}' initialized.".format(config['architecture']))
        """
        elif config['architecture'] == "tsainfcnnrnnlex":
            #unnecessary paremters, to be removed
            x1_size = 48000 # 120 * 400
            x1_channels = 1;
            x2_size = 360   # 120 * 3
            x2_height = 3   # number of lexicons
            x2_channels = 1;
            input_channels = 1
            filter2_sizes = [3,4,5]
            output2_channels_1 = 64
            output_channels_2 = 64
            dense_output_1 = 128
            # size of output I guess from rnn cell for word embeddings
            num_hidden1 = 256
        """

        write_summaries = True
        if config["cross_valid"] != 0:
            # Turn off summaries for cross validation as they are not working then
            write_summaries = False
            if verbose >= 2:
                print("Turned off summaries for cross validation to avoid errors.")

        # Set loss and optimizer
        learning_rate = tf.get_variable("learning_rate", dtype=tf.float32, initializer=tf.constant(config["learning_rate"]))
        if config["optimizer"] == "adam":
            opt = tf.train.AdamOptimizer(learning_rate, beta2=config["adam_b2"])
            train_op = opt.minimize(model.cross_entropy)
        else:
            raise ValueError("Invalid optimizer parameter value {0:s}".format(config["optimizer"]))
        tf.add_to_collection("train_op", train_op)

        # Init session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if verbose >= 2:
            print("Current name scope: {0:s}".format(tf.get_variable_scope().name))
        # Init model saver
        saver = tf.train.Saver(max_to_keep=20)

        # Setup summaries
        if write_summaries:
            tf.summary.scalar("loss", model.cross_entropy)
            summary_op = tf.summary.merge_all()
            # Setup summary writers
            out_dir = os.path.join(os.path.abspath(os.path.curdir), MODEL_OUTPUT_PATH, config['run_name'])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)
            best_conf_mat, best_epoch, best_eval_metric = _train_run(config, sess, model, train_op, saver, x_dev, x_train, y_dev, y_train,
                                                   write_summaries, summary_op, train_summary_writer, dev_summary_writer,
                                                   verbose)
        else:
            best_conf_mat, best_epoch, best_eval_metric = _train_run(config, sess, model, train_op, saver, x_dev, x_train, y_dev, y_train,
                                                   verbose=verbose)

        sess.close()

        # In case annealing restarts are set, anneal learn rate by division of config["annealing_factor"]
        # http://ruder.io/deep-learning-nlp-best-practices/index.html#optimization
        if config["annealing_restarts"] > 0:
            # Create output lists for returning all anneal runs
            best_conf_mat = [best_conf_mat]
            best_eval_metric = [best_eval_metric]
            best_epoch = [best_epoch]
            model = [model]
        annealing_learn_rate = config["learning_rate"]
        anneal_run = 1
        annealing_restarts = config["annealing_restarts"]
        while annealing_restarts > 0:
            annealing_learn_rate = annealing_learn_rate / config["annealing_factor"]
            print("Anneal run {0:d}, new learning rate: {1:f}".format(anneal_run, annealing_learn_rate))

            config["learning_rate"] = annealing_learn_rate
            #config['run_name'] = config['run_name'] + "_a" + str(anneal_run)
            checkpoint_file = get_model_path(config['run_name'], "best_model")
            #with tf.variable_scope("anneal_" + str(anneal_run)):
            best_conf_mat_an, best_epoch_an, best_eval_metric_an, model_an = train_saved_model(
                config,
                checkpoint_file,
                x_train,
                y_train,
                x_dev,
                y_dev,
                k_fold_id=k_fold_id,
                verbose=verbose
            )
            best_conf_mat.append(best_conf_mat_an)
            best_epoch.append(best_epoch_an)
            best_eval_metric.append(best_eval_metric_an)
            model.append(model_an)
            annealing_restarts -= 1
            anneal_run += 1

    return best_conf_mat, best_epoch, best_eval_metric, model


def train_saved_model(config, checkpoint_file, x_train, y_train, x_dev, y_dev, k_fold_id=None, verbose=0):
    """
    Load saved model from checkpoint file and continue training.

    Allows changed learning rate, rest stays the same.

    :param config:
    :param checkpoint_file:
    :param x_train:
    :param y_train:
    :param x_dev:
    :param y_dev:
    :param verbose:
    :return:
    """
    write_summaries = True
    if config["cross_valid"] != 0:
        # Turn off summaries for cross validation as they are not working then
        write_summaries = False
        if verbose >= 2:
            print("Turned off summaries for cross validation to avoid errors.")

    sess = tf.Session()

    if k_fold_id is not None:
        variable_scope = "k_fold_" + str(k_fold_id)
    else:
        variable_scope = ""

    tf.get_variable_scope().reuse_variables()

    # Load model
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    if verbose >= 2:
        print("Current name scope: {0:s}".format(tf.get_variable_scope().name))
    if verbose >= 2:
        print("Initializing model: " + config['architecture'] + "...")

    if config['architecture'] == "tsainfcnn":
        model = TsaInfCnnMapping(sess.graph, variable_scope=variable_scope)
    elif config['architecture'] == "attention-cnn":
        model = AttentionCNNMapping(sess.graph, variable_scope=variable_scope)
    else:
        print("ERROR: unrecognized architecture " + config['architecture'])
    if verbose >= 1:
        print("Model '{0:s}' initialized.".format(config['architecture']))

    learning_rate = tf.get_variable("learning_rate", dtype=tf.float32, initializer=tf.constant([config["learning_rate"]]))
    # Update learning rate
    lr_assign_op = learning_rate.assign(config["learning_rate"])
    sess.run(lr_assign_op)
    if k_fold_id is not None:
        train_op = tf.get_collection("train_op")[0]
    else:
        train_op = tf.get_collection("train_op", scope=variable_scope)[0]

    # TODO: fix summaries
    write_summaries = False
    # Setup summaries
    if write_summaries:
        tf.summary.scalar("loss", model.cross_entropy)
        summary_op = tf.summary.merge_all()
        # Setup summary writers
        out_dir = os.path.join(os.path.abspath(os.path.curdir), MODEL_OUTPUT_PATH, config['run_name'])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)
        best_conf_mat, best_epoch, best_eval_metric = _train_run(config, sess, model, train_op, saver, x_dev, x_train, y_dev, y_train,
                                               write_summaries, train_summary_writer, summary_op, dev_summary_writer,
                                               verbose)
    else:
        best_conf_mat, best_epoch, best_eval_metric = _train_run(config, sess, model, train_op, saver, x_dev, x_train, y_dev, y_train,
                                               verbose=verbose)

    sess.close()
    return best_conf_mat, best_epoch, best_eval_metric, model


def _train_run(config, sess, model, train_op, saver, x_dev, x_train, y_dev, y_train, write_summaries=False,
               summary_op=None, train_summary_writer=None, dev_summary_writer=None, verbose=0):
    """
    Run training over defined epochs or until early stopping.

    :param config:
    :param sess:
    :param model:
    :param train_op:
    :param saver:
    :param x_dev:
    :param x_train:
    :param y_dev:
    :param y_train:
    :param write_summaries:
    :param summary_op:
    :param train_summary_writer:
    :param dev_summary_writer:
    :param verbose:
    :return:
    """
    total_time = 0
    best_eval_metric = -1
    best_loss = sys.float_info.max
    best_epoch = 0
    best_conf_mat = None
    # Log no improvement epochs for early stopping
    no_improvement_epochs = 0
    step = 0
    for epoch_id in range(config['num_epochs']):
        # Split batches, In epoch loop, so number of epochs set to 1
        batches_train = data_helper.batch_iter(list(zip(x_train, y_train)), config['batch_size'], 1)

        time1 = time.time()
        train_losses = []
        for batch in batches_train:
            batch_xs, batch_ys = zip(*batch)

            train_dict = {
                model.x: batch_xs,
                model.y_: batch_ys,
                model.keep_prob: config['keep_prob']
            }
            if write_summaries:
                _, step_loss, summaries = sess.run([train_op, model.cross_entropy, summary_op], feed_dict=train_dict)
                # Log loss summary
                train_summary_writer.add_summary(summaries, step)
            else:
                _, step_loss = sess.run([train_op, model.cross_entropy], feed_dict=train_dict)

            train_losses.append(step_loss)

            if verbose >= 2:
                # Print step details
                time_str = datetime.datetime.now().isoformat()
                print("{0:s}: {1:s} step {2:7d}, loss {3:7.6f}".format("TRN", time_str, step, step_loss))

            step += 1
        # Calculate epoch loss
        avrg_trn_loss = sum(train_losses) / float(len(train_losses))
        if verbose >= 1:
            # Print step details
            time_str = datetime.datetime.now().isoformat()
            print("{0:s}: {1:s} epoch {2:3d}, loss {3:7.6f}".format("TRN", time_str, epoch_id, avrg_trn_loss))

        time2 = time.time()
        total_time = total_time + time2 - time1
        if verbose >= 2:
            print("Avg Training Time per Iteration (seconds): " + str(total_time / (epoch_id + 1)))

        # Evaluate on dev data
        if write_summaries:
            dev_loss, eval_metric, conf_mat, _ = eval_step(
                model,
                sess,
                x_dev,
                y_dev,
                epoch_id=epoch_id,
                summary_op=summary_op,
                dev_summary_writer=dev_summary_writer,
                step=step,
                verbose=verbose
            )
        else:
            dev_loss, eval_metric, conf_mat, _ = eval_step(
                model,
                sess,
                x_dev,
                y_dev,
                eval_metric=config["eval_metric"],
                eval_labels=config["eval_labels"],
                epoch_id=epoch_id,
                verbose=verbose
            )

        if config["eval_metric"] == "loss":
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch_id
                best_conf_mat = conf_mat
                save_model(config, saver, sess, "best_model", epoch_id)
                # Reset no improvement for early stopping
                no_improvement_epochs = 0
            elif not math.isnan(dev_loss):
                # Log no improvement for early stopping
                no_improvement_epochs += 1
        else:
            if best_eval_metric < eval_metric:
                best_eval_metric = eval_metric
                best_epoch = epoch_id
                best_conf_mat = conf_mat
                save_model(config, saver, sess, "best_model", epoch_id)
                # Reset no improvement for early stopping
                no_improvement_epochs = 0
            elif not math.isnan(eval_metric):
                # Log no improvement for early stopping
                no_improvement_epochs += 1

        if config['save_iteration'] != 0 and epoch_id + 1 % config['save_iteration'] == 0:
            msg = "epoch_id_" + str(epoch_id)
            save_model(config, saver, sess, msg, epoch_id)

        if config['early_stop_patience'] is not None:
            if no_improvement_epochs >= config['early_stop_patience']:
                # Early stopping patience exceeded
                print("Early stopping patience exceeded in epoch {0:d} after no improvement over {1:d} epochs"
                      .format(epoch_id, no_improvement_epochs))
                break
    return best_conf_mat, best_epoch, best_eval_metric


def save_model(config,saver,sess,msg, iter):
    # print("Saving "+msg+" model so far")
    start = time.time()
    path = get_model_path(config['run_name'], msg)
    # path = os.path.join(os.path.abspath(os.path.curdir), MODEL_OUTPUT_PATH, config['run_name'], msg, "model.ckpt")
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = saver.save(sess, path)
    end = time.time()
    # print(msg + " Model saved (" + iter + ") in file: %s" % save_path)
    #  print("Save time: " + str(end-start) + " seconds")
    return None


def get_model_path(run_name, msg):
    """
    Join current dir, MODEL_OUTPUT_PATH, run name, msg and 'model.ckpt' to a path.

    :param run_name: string
        Experiment identifier
    :param msg: string
        Additional identifier
    :return: string
        Path to model as defined by inptu
    """
    path = os.path.join(os.path.abspath(os.path.curdir), MODEL_OUTPUT_PATH, run_name, msg, "model.ckpt")
    return path


def eval_step(model, sess, x_input, y_input, batch_size=None, epoch_id=0, summary_op=None, dev_summary_writer=None,
              eval_metric="f1_macro", eval_labels=None, step=None, verbose=0):
    """

    :param model:
    :param sess:
    :param x_input:
    :param y_input:
    :param batch_size:
    :param epoch_id:
    :param summary_op:
    :param dev_summary_writer:
    :param eval_metric: string
        Identifier for eval metric {f1_macro, f1_micro}
    :param eval_labels: array
        Sett of labels to include. For multilabel targets, labels are column indices. By default (None), all labels in
        y_true and y_pred are used in sorted order.
    :param step:
    :param verbose:
    :return:
    """

    # Collect the predictions, gold labels here
    all_predictions = []
    all_scores = []

    if batch_size is None:
        batch_eval = False
    else:
        batch_eval = True

    y_gold = numpy.argmax(y_input, 1)

    if batch_eval:
        # Split batches, In epoch loop, so number of epochs set to 1
        batches_test = data_helper.batch_iter(list(zip(x_input, y_input)), config['batch_size'], 1)

        for test_batch in batches_test:
            test_batch_xs, test_batch_ys = zip(*test_batch)

            # keep prob must be one
            eval_dict = {
                model.x: test_batch_xs,
                model.y_: test_batch_ys,
                model.keep_prob: 1.0
            }

            if summary_op is not None:
                batch_predictions, batch_scores, step_loss, summaries = sess.run(
                    [model.preds, model.y_conv, model.cross_entropy, summary_op],
                    feed_dict=eval_dict
                )
                # Log loss summary
                dev_summary_writer.add_summary(summaries, step)
            else:
                batch_predictions, batch_scores, step_loss = sess.run(
                    [model.preds, model.y_conv, model.cross_entropy],
                    feed_dict=eval_dict
                )

            # Concatenate predictions and gold for evaluation
            all_predictions = numpy.concatenate([all_predictions, batch_predictions])
            all_scores = numpy.concatenate([all_scores, batch_scores])
            y_gold = numpy.concatenate([y_gold, numpy.argmax(test_batch_ys, 1)])

        eval_metric = eval.get_eval_metric(y_gold, all_predictions, eval_metric, eval_labels)

    else:
        eval_dict = {
            model.x: x_input,
            model.y_: y_input,
            model.keep_prob: 1.0
        }

        if summary_op is not None:
            all_predictions, all_scores, loss, summaries = sess.run(
                [model.preds, model.y_conv, model.cross_entropy, summary_op],
                feed_dict=eval_dict
            )
            # Log loss summary
            dev_summary_writer.add_summary(summaries, step)
        else:
            all_predictions, all_scores, loss = sess.run(
                [model.preds, model.y_conv, model.cross_entropy],
                feed_dict=eval_dict
            )

        eval_metric = eval.get_eval_metric(y_gold, all_predictions, eval_metric, eval_labels)

        # Print step details
        if verbose >= 1:
            time_str = datetime.datetime.now().isoformat()
            print("{0:s}: {1:s} epoch {2:3d}, loss {3:7.6f}, Score {4:7.4f}"
                  .format("DEV", time_str, epoch_id, loss, eval_metric))

    if verbose >= 2:
        eval.print_results(y_gold, all_predictions)

    conf_mat = eval.get_confustion_matrix(y_gold, all_predictions)
    return loss, eval_metric, conf_mat, all_scores


def full_train(config, verbose=False):
    """
    Load data and initialize training.

    :param config:
    :param verbose:
    :return:
    """

    if config['dev_path'] is None and config['cross_valid'] != 0:
        #cross validation
        split_data = data_helper.load_data_split_npy(config['train_path'], config['n_classes'], config['cross_valid'], one_hot=True)
        f1_scores = []
        for i in range(config['cross_valid']):
        #ids k
            ids = split_data[i][0]
        #labels k
            #dev_data = split_data[i][1]
            dev_basic = numpy.array(split_data[i][1], dtype=int)
            dev_basic = dev_basic - numpy.min(dev_basic, axis=0)
            dev_labels = numpy.eye(config['n_classes'], dtype=int)[dev_basic]
            x,y,z = numpy.shape(dev_labels)
            dev_labels = numpy.array(dev_labels).reshape(x,z)
        #data k
            dev_data = split_data[i][2]
            #train_data=numpy.zeros([1,args.embedding_dimension*args.document_length])
            #train_labels=numpy.zeros([1,1])
            #for j in range(args.cross_valid):
            #    if j==i:
            #        continue
            #    else:
            #        train_data = numpy.vstack((train_data,split_data[j][2]))
            #        train_labels = numpy.vstack((train_labels,split_data[j][1]))
            #train_data = train_data[1:,:]
            #train_labels = train_labels[1:,:]
            #train_labels = train_labels[1:,:]
            rows,cols = numpy.shape(ids)
            train_data = numpy.empty([(config['cross_valid']-1)*rows,config['embedding_dimension']*config['document_length']])
            train_labels= numpy.empty([(config['cross_valid']-1)*rows,1])
            count=0
            for j in range(config['cross_valid']):
                if j==i:
                    continue
                else:
                    train_data[count*rows:(count+1)*rows,:]=split_data[j][2]
                    train_labels[count*rows:(count+1)*rows,:]=split_data[j][1]
                    count+=1
            train_basic = numpy.array(train_labels, dtype=int)
            train_basic = train_basic - numpy.min(train_basic, axis=0)
            train_labels = numpy.eye(config['n_classes'], dtype=int)[train_basic]
            x,y,z = numpy.shape(train_labels)
            train_labels = numpy.array(train_labels).reshape(x,z)
            #print(numpy.shape(train_labels))
          #  print("running cross_validation fold" + str(i))
            """
            #K-fold specific variable scope, so model variables don't get resused
            with tf.variable_scope("k_fold_" + str(i)):
                best_conf_mat_k, best_epoch_k, _ = train(config, train_data, train_labels, dev_data, dev_labels,
                                                         k_fold_id=i, verbose=verbose)
                best_f1_k = eval.conf_mat_f1(best_conf_mat_k)
            """
            # Assign unique name to kfold experiment
            k_fold_config = config.copy()
            k_fold_config["run_name"] = cross_valid_run_name(config["run_name"], i)
            # Run training
            if config["annealing_restarts"] > 0:
                best_conf_mat_k_list, _, best_eval_metric, _ = train(k_fold_config, train_data, train_labels, dev_data, dev_labels,
                                                   verbose=verbose)
                #best_f1_k_list = [eval.conf_mat_f1(best_conf_mat_k) for best_conf_mat_k in best_conf_mat_k_list]
                f1_scores.append(best_eval_metric)
            else:
                best_conf_mat_k, _, best_eval_metric, _ = train(k_fold_config, train_data, train_labels, dev_data, dev_labels,
                                              verbose=verbose)
                #best_f1_k = eval.conf_mat_f1(best_conf_mat_k)
                f1_scores.append(best_eval_metric)
        # Calculate mean and std dev
        if config["annealing_restarts"] > 0:
            f1_scores = numpy.asarray(f1_scores)
            mean = numpy.mean(f1_scores, axis=0)
            std = numpy.std(f1_scores, axis=0)
            for i in range(config["annealing_restarts"]):
                print("The mean eval score (anneal {0:d}) is: {1:s}".format(i, str(mean[i])))
                # Add the print statement of printing the config
                print("The standard deviation (anneal {0:d}) is: {1:s}".format(i, str(std[i])))
            print("The mean eval score is: "+str(mean[config["annealing_restarts"]]))
            # Add the print statement of printing the config
            print("The standard deviation is: "+str(std[config["annealing_restarts"]]))
        else:
            f1_scores = numpy.asarray(f1_scores)
            mean = numpy.mean(f1_scores)
            std = numpy.std(f1_scores)
            print("The mean eval score is: "+str(mean))
            # Add the print statement of printing the config
            print("The standard deviation is: "+str(std))
    elif config['cross_valid'] is 0 and config['dev_path'] is not None:
        #not cross validation
        #Load numpy data
        start = time.time()
       # print("Loading Data, this might take a while....")
        if config['lexicon_embedding_dimension'] == 0:
            x_train, y_train, _ = data_helper.load_data_npy(config['train_path'], config['n_classes'], config['one_hot'])
        else:
            x_train, x_train_lex, y_train, _ = data_helper.load_data_lex_npy(config['train_path'], config['n_classes'], config['lexicon_embedding_dimension']*config['document_length'])
        end = time.time()
      #  print("Loaded " + str(len(x_train)) + " dimension ( " + str(len(x_train[0])) + " ) train data items in " + str(end-start) + " seconds")

        start = end
        if config['lexicon_embedding_dimension'] == 0:
            x_dev, y_dev, _ = data_helper.load_data_npy(config['dev_path'], config['n_classes'], config['one_hot'])
        else:
            x_dev, x_dev_lex, y_dev, _ = data_helper.load_data_lex_npy(config['dev_path'], config['n_classes'], config['lexicon_embedding_dimension']*config['document_length'])
        end = time.time()
     #   print("Loaded " + str(len(x_dev)) + " dimension ( " + str(len(x_dev[0])) + " ) dev data items in " + str(end-start) + " seconds")
        if config["annealing_restarts"] > 0:
            best_conf_mat_list, _, best_eval_metric, _ = train(config, x_train, y_train, x_dev, y_dev, verbose=verbose)
            best_f1 = [eval.conf_mat_f1(best_conf_mat) for best_conf_mat in best_conf_mat_list]
            best_recall = [eval.conf_mat_recall(best_conf_mat) for best_conf_mat in best_conf_mat_list]
            best_precision = [eval.conf_mat_precision(best_conf_mat) for best_conf_mat in best_conf_mat_list]
            for i in range(config["annealing_restarts"]):
                print("Avrg F1 score (anneal {0:d}) is: {1:f}".format(i, best_f1[i]))
                print("Avrg Racall (anneal {0:d}) is: {1:f}".format(i, best_recall[i]))
                print("Avrg Precision (anneal {0:d}) is: {1:f}".format(i, best_precision[i]))
            print("Avrg F1 score is : " + str(best_f1[config["annealing_restarts"]]))
            print("Avrg Racall is : " + str(best_recall[config["annealing_restarts"]]))
            print("Avrg Precision is : " + str(best_precision[config["annealing_restarts"]]))
            print("Best {0:s} score is : {1:s}".format(config["eval_metric"], str(best_eval_metric)))
        else:
            best_conf_mat, _, best_eval_metric, _ = train(config, x_train, y_train, x_dev, y_dev, verbose=verbose)
            best_f1 = eval.conf_mat_f1(best_conf_mat)
            best_recall = eval.conf_mat_recall(best_conf_mat)
            best_precision = eval.conf_mat_precision(best_conf_mat)
            print("Avrg F1 score is : " + str(best_f1))
            print("Avrg Racall is : " + str(best_recall))
            print("Avrg Precision is : " + str(best_precision))
            print("Best {0:s} score is : {1:s}".format(config["eval_metric"], str(best_eval_metric)))


        """
        if config['test_path'] is not None:
            start = end
            if config['lexicon_embedding_dimension'] == 0:
                x_test, y_test, _ = data_helper.load_data_npy(config['test_path'], config['n_classes'], config['one_hot'])
            else:
                x_test, x_test_lex, y_test, _ = data_helper.load_data_lex_npy(config['test_path'], config['n_classes'], config['lexicon_embedding_dimension']*config['document_length'])
                end = time.time()
                print("Loaded " + str(len(x_test)) + " dimension ( " + str(len(x_test[0])) + " ) test data items in " + str(end-start) + " seconds")
            avg_f1, conf_mat, _, _ = eval_step(model, sess, x_dev, y_dev, verbose)
        """
      
    return None


def cross_valid_run_name(run_name, cross_valid_id):
    """
    Appends cross valid identifier and id to input run name.

    :param run_name: string
        Original run name
    :param cross_valid_id: int
        Id of cross valid run
    :return: string
        run_name + "_cv" + str(i)
    """
    return run_name + "_cv" + str(cross_valid_id)


if __name__ == "__main__":
    # List for high level param, that don't go into config objects
    high_level_param = []
    train_parser = argparse.ArgumentParser(add_help=False)
    # Experiment parameters
    train_parser.add_argument('--run_name', '-r', required=True,
                              help="Name of training run, determines the model name and checkpoint location.")
    train_parser.add_argument('--save_iteration', type=int, default=0,
                              help="Aside from best model save each save_iterations-th model (default: 0, off)")
    train_parser.add_argument('--cross_valid', type=int, default=0, help="value of k for cross-validation")
    train_parser.add_argument('--early_stop_patience', type=int, default=3,
                              help="Number of epochs without improvement after which to stop training (default: 3)")
    train_parser.add_argument('--eval_metric', default="f1_macro",
                              help="Metric for saving best model and evaluating early stop "
                                   "{loss, f1_macro, f1_micro, recall_macro} (default: f1_macro)")
    train_parser.add_argument('--eval_labels', type=int, nargs='+', default=None,
                              help="Labels for eval metric, None means all (default: None)")
    train_parser.add_argument('--verbose', '-v', type=int, default=0,
                              help="Set for more verbose output {0 - off, 1 - some, 2 -all} (default: 0).")
    high_level_param.append("verbose")
    # Data parameters
    train_parser.add_argument('--document_length', '-d', type=int, default=120,
                              help="Number of input tokens per document (default: 120)")
    train_parser.add_argument('--embedding_dimension', '-e', type=int, required=True,
                              help="Dimension of input word embeddings")
    train_parser.add_argument('--n_classes', '-c', type=int, required=True, help="Number of classes in data")
    train_parser.add_argument('--lexicon_embedding_dimension', type=int, default=0,
                              help="Dimension of input lexicon dimensions (default: 0, ignore)")
    train_parser.add_argument('--train_path', '-trn', required=True, help="Path to training data")
    train_parser.add_argument('--dev_path', '-dev', help="Path to development/validation data")
    train_parser.add_argument('--test_path', '-tst', help="Path to testing data")
    train_parser.add_argument('--no_one_hot', dest='one_hot', action='store_false',
                              help="Set if labels in input data are not one-hot encoded")
    # Hyper-parameters
    train_parser.add_argument('--learning_rate', '-l', type=float, default=0.0001,
                              help="Learning rate (default: 0.0001)")
    train_parser.add_argument('--num_epochs', '-n', type=int, default=5,
                              help="Number of epochs over full training data (default: 5)")
    train_parser.add_argument('--optimizer', default="adam",
                              help="Optimizer for backpropagation {adam} (default: adam)")
    train_parser.add_argument('--adam_b2', type=float, default=0.999,
                              help="Adam decay rate for l2 norm moving average (default: 0.999)")
    train_parser.add_argument('--annealing_restarts', type=int, default=2,
                              help="Number of learning rate annealing restarts {0 - off} (default: 2)")
    train_parser.add_argument('--annealing_factor', type=int, default=2,
                              help="Division factor for annealing learning rate (default: 2)")
    train_parser.add_argument('--architecture', '-a', required=True, default="tsainfcnn",
                              help="select the deep learning architecture, options = {tsainfcnn} (default: tsainfcnn)")
    train_parser.add_argument('--batch_size', '-b', type=int, default=100,
                              help="Size of training and eval batches (default: 100)")
    train_parser.add_argument('--keep_prob', type=float, default=0.7,
                              help="Keep probability for dropout (default: 0.7)")
    # Architecture specific Hyper-parameters
    if "tsainfcnn" in sys.argv:
        tsainfcnn_parser = argparse.ArgumentParser(parents=[train_parser])
        tsainfcnn_parser.add_argument('--embedd_filter_sizes', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                                      help="Filter dimensions for word embedding CNN convolutions "
                                           "(default: [1,2,3,4,5])")
        tsainfcnn_parser.add_argument('--n_filters', type=int, default=256,
                                      help="Number of convolutions per filter size (default: 256)")
        tsainfcnn_parser.add_argument('--n_dense_output', type=int, default=256,
                                      help="Number of units in dense layer (default: 256)")
        args = tsainfcnn_parser.parse_args()
    elif "attention-cnn" in sys.argv:
        attention_cnn_parser = argparse.ArgumentParser(parents=[train_parser])
        attention_cnn_parser.add_argument('--embedd_filter_sizes', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                                          help="Filter dimensions for word embedding CNN convolutions "
                                               "(default: [1,2,3,4,5])")
        attention_cnn_parser.add_argument('--n_filters', type=int, default=256,
                                          help="Number of convolutions per filter size (default: 256)")
        attention_cnn_parser.add_argument('--n_dense_output', type=int, default=256,
                                          help="Number of units in dense layer (default: 256)")
        attention_cnn_parser.add_argument('--attention_depth', type=int, default=50,
                                          help="Depth of attention vector (default: 50)")
        args = attention_cnn_parser.parse_args()
    else:
        raise ValueError("Could not identifier architecture argument.")

    config = {}
    for arg in vars(args):
        print(arg + ': ' + str(getattr(args, arg)))
        if arg not in high_level_param:
            config[arg] = getattr(args, arg)
        else:
            if DEBUG:
                print("DEBUG: High level param '{0:s}' not added to config'".format(arg))

    full_train(config, args.verbose)
