# TensorFlow Text Classification

Technically this project is designed for classification with arbitrary Tensorflow architectures, practically currently only a short document (sentence) CNN architecture is implemented.

### Architectures

#### TSA-INF-CNN

[master/tsa_inf_cnn.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/tsa_inf_cnn.py)

One layer Convolutional Neural Net with single dense layer, expecting pre-trained word embeddings.

Based on [semeval submission 2017 ](http://nlp.arizona.edu/SemEval-2017/pdf/SemEval135.pdf) CNN model, where it outperformed submitted ensemble models on positive negative recall.

Recommended for text documents of tweet or abstract size.

#### attention_cnn (experimental)

[master/attention_cnn.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/attention_cnn.py)

TSA-INF-CNN architecture with attention. Attention either doesn't show improvement in results or isn't work. Attention needs to be visualized with meaningful example, e.g. sentiment analysis.

### Requirements

```
tensorflow==1.1.0
numpy==1.12.1
```

[Data-Wrangler](http://10.188.49.222:3000/jasper/Data-Wrangler)

Confirmed to run in "tensorflow3" anaconda environment on 100 - *Sept 21, 2017*

Confirmed to run in standard python environment on 101 - *Sept 19, 2017*

## Training

[master/train.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/train.py)

Train a selected architecture with train/dev split or cross-validation.

##### Parameters

```
  --run_name RUN_NAME, -r RUN_NAME
                        Name of training run, determines the model name and
                        checkpoint location.
  --save_iteration SAVE_ITERATION
                        Aside from best model save each save_iterations-th
                        model (default: 0, off)
  --cross_valid CROSS_VALID
                        value of k for cross-validation
  --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs without improvement after which to
                        stop training (default: 3)
  --eval_metric EVAL_METRIC
                        Metric for saving best model and evaluating early stop
                        {loss, f1_macro, f1_micro, recall_macro} (default:
                        f1_macro)
  --eval_labels EVAL_LABELS [EVAL_LABELS ...]
                        Labels for eval metric, None means all (default: None)
  --verbose VERBOSE, -v VERBOSE
                        Set for more verbose output {0 - off, 1 - some, 2
                        -all} (default: 0).
  --document_length DOCUMENT_LENGTH, -d DOCUMENT_LENGTH
                        Number of input tokens per document (default: 120)
  --embedding_dimension EMBEDDING_DIMENSION, -e EMBEDDING_DIMENSION
                        Dimension of input word embeddings
  --n_classes N_CLASSES, -c N_CLASSES
                        Number of classes in data
  --lexicon_embedding_dimension LEXICON_EMBEDDING_DIMENSION
                        Dimension of input lexicon dimensions (default: 0,
                        ignore)
  --train_path TRAIN_PATH, -trn TRAIN_PATH
                        Path to training data
  --dev_path DEV_PATH, -dev DEV_PATH
                        Path to development/validation data
  --test_path TEST_PATH, -tst TEST_PATH
                        Path to testing data
  --no_one_hot          Set if labels in input data are not one-hot encoded
  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate (default: 0.0001)
  --num_epochs NUM_EPOCHS, -n NUM_EPOCHS
                        Number of epochs over full training data (default: 5)
  --optimizer OPTIMIZER
                        Optimizer for backpropagation {adam} (default: adam)
  --adam_b2 ADAM_B2     Adam decay rate for l2 norm moving average (default:
                        0.999)
  --annealing_restarts ANNEALING_RESTARTS
                        Number of learning rate annealing restarts {0 - off}
                        (default: 2)
  --annealing_factor ANNEALING_FACTOR
                        Division factor for annealing learning rate (default:
                        2)
  --architecture ARCHITECTURE, -a ARCHITECTURE
                        select the deep learning architecture, options =
                        {tsainfcnn} (default: tsainfcnn)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Size of training and eval batches (default: 100)
  --keep_prob KEEP_PROB
                        Keep probability for dropout (default: 0.7)
```


##### Command line example

(see [master/train_cmd.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/train_cmd.txt) for more examples)

```
nohup \
python train.py \
--architecture tsainfcnn \
--run_name cross_val_test_anneal \
--learning_rate 0.0001 \
--num_epochs 5 \
--batch_size 100 \
--document_length 47 \
--embedding_dimension 400 \
--n_classes 3 \
--train_path /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_cv_jinho_400.npy \
--cross_valid 5 \
--no_one_hot \
--keep_prob 0.7 \
--embedd_filter_sizes 2 3 4 \
--n_filters 100 \
--n_dense_output 100 \
--adam_b2 0.9 \
--verbose 2 \
> cross_val_test_anneal.out 
```

## Evaluation

[master/decode.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/decode.py)

Load an existing model and evaluate test data input. This serves as an example of how to decode with stored model.

##### Parameters

```
  --verbose VERBOSE, -v VERBOSE
                        Set for more verbose output {0 - off, 1 - some, 2
                        -all} (default: 0).
  --eval_metric EVAL_METRIC
                        Metric for saving best model and evaluating early stop
                        {loss, f1_macro, f1_micro, recall_macro} (default:
                        f1_macro)
  --eval_labels EVAL_LABELS [EVAL_LABELS ...]
                        Labels for eval metric, None means all (default: None)
  --experiment_eval     Set to evaluate all experiments defined by --run_name
                        (default: False)
  --submission_out SUBMISSION_OUT
                        Output path for test file with prediction annotation.
  --submission_in SUBMISSION_IN
                        Input path for submission file, system will append ' '
                        followed by test_path label.
  --test_path TEST_PATH [TEST_PATH ...], -tst TEST_PATH [TEST_PATH ...]
                        Path to evaluation data
  --input_type INPUT_TYPE
                        Type of input data {'npy', 'txt', 'npy_lex'} (default:
                        'npy')
  --embedd_model_path EMBEDD_MODEL_PATH
                        Path to embedding model, only required if
                        input_type='txt'
  --no_one_hot
  --document_length DOCUMENT_LENGTH, -d DOCUMENT_LENGTH
                        Number of input tokens per document (default: 120)
  --embedding_dimension EMBEDDING_DIMENSION, -e EMBEDDING_DIMENSION
                        Dimension of input word embeddings
  --n_classes N_CLASSES, -c N_CLASSES
                        Number of classes in data
  --lexicon_embedding_dimension LEXICON_EMBEDDING_DIMENSION
                        Dimension of input lexicon dimensions (default: 0,
                        ignore)
  --architecture ARCHITECTURE [ARCHITECTURE ...], -a ARCHITECTURE [ARCHITECTURE ...]
                        select the deep learning architecture, options =
                        {tsainfcnn} (default: [tsainfcnn])
  --run_name RUN_NAME [RUN_NAME ...], -r RUN_NAME [RUN_NAME ...]
                        Name of training run, determines the model name and
                        checkpoint location.
  --run_suffix RUN_SUFFIX [RUN_SUFFIX ...]
                        Suffix for identifying checkpoint of multiple saved
                        models (default: ['best_model']
  --checkpoint_file CHECKPOINT_FILE [CHECKPOINT_FILE ...]
                        Path to checkpoint file from training run, alternative
                        to run_name
  --variable_scope VARIABLE_SCOPE [VARIABLE_SCOPE ...]
                        Additional variable scope for loading model parameters
                        (default: [''])
  --cross_valid CROSS_VALID [CROSS_VALID ...]
                        Value of k for cross-validation (default: [0])
```

##### Parameters - tsainfcnn

```
  --embedd_filter_sizes EMBEDD_FILTER_SIZES [EMBEDD_FILTER_SIZES ...]
                        Filter dimensions for word embedding CNN convolutions
                        (default: [1,2,3,4,5])
  --n_filters N_FILTERS
                        Number of convolutions per filter size (default: 256)
  --n_dense_output N_DENSE_OUTPUT
                        Number of units in dense layer (default: 256)
```

##### Command line Example

(see [master/decode_cmd.txt](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/master/decode_cmd.txt) for more examples)

```
nohup \
python \
decode.py \
--architecture tsainfcnn \
--run_name \
smm4h_task2_all_godin_wide_filter_0_1772 \
smm4h_task2_all_godin_wide_filter_0_3866 \
smm4h_task2_all_godin_wide_filter_0_4599 \
smm4h_task2_all_godin_wide_filter_0_77 \
smm4h_task2_all_jinho_wide_filter_0_133 \
--cross_valid 5 \
--test_path /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_all_test_jinho_400.npy /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_all_test_godin_400.npy \
--no_one_hot \
--document_length 47 \
--embedding_dimension 400 \
--n_classes 3 \
--verbose 1 \
--eval_metric f1_micro \
--eval_labels 0 1 \
> smm4h_task2_eval_top5.out
```

## Self-Service Training

[jasper-local/automated_train.py](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/jasper-local/automated_train.py)

Runs random search on input parameter ranges. All permutations of parameter ranges are generated and a random subset of possible permutations is used as training configurations and evaluated. Random search is terminated after max time input is exceeded. Self-service training can be resumed with exactly the same input (changing input, changes permutations and thus experiment ids). With a reasonable choice of parameter ranges and sufficient time this should produce results at least on par with manual parameter tuning.

##### Parameters

```

 
  --max_running_time MAX_RUNNING_TIME
                        Maximum training time in hours, script will terminate
                        when exceeded (default: 12)
  --no_shuffle_configs  Set to turn off random shuffle of parameter
                        permutations, highly discouraged unless all
                        permutations are run
  --experiment_name EXPERIMENT_NAME, -r EXPERIMENT_NAME
                        Name of self-service training run, determines the
                        model names and checkpoint locations
  --eval_metric EVAL_METRIC
                        Metric for saving best model and evaluating early stop
                        {loss, f1_macro, f1_micro, recall_macro} (default:
                        f1_macro)
  --eval_labels EVAL_LABELS [EVAL_LABELS ...]
                        Labels for eval metric, None means all (default: None)
  --verbose VERBOSE, -v VERBOSE
                        Set for more verbose output {0 - off, 1 - some, 2
                        -all} (default: 0).
  --train_path TRAIN_PATH, -trn TRAIN_PATH
                        Path to training data
  --dev_path DEV_PATH, -dev DEV_PATH
                        Path to development/validation data
  --test_path TEST_PATH, -tst TEST_PATH
                        Path to testing data
  --document_length DOCUMENT_LENGTH, -d DOCUMENT_LENGTH
                        Number of input tokens per document (default: 120)
  --embedding_dimension EMBEDDING_DIMENSION, -e EMBEDDING_DIMENSION
                        Dimension of input word embeddings
  --lexicon_embedding_dimension LEXICON_EMBEDDING_DIMENSION
                        Dimension of input lexicon dimensions (default: 0,
                        ignore)
  --n_classes N_CLASSES, -c N_CLASSES
                        Number of classes in data, determines output dimension
  --no_one_hot          Set if labels in input data are not one-hot encoded
  --cross_valid CROSS_VALID
                        value of k for cross-validation (default: 0, train/dev
                        split)
  --save_iteration SAVE_ITERATION
                        Aside from best model save each save_iterations-th
                        model (default 0, off)
  --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs without improvement after which to
                        stop training (default: 3)
  --optimizer OPTIMIZER
                        Optimizer for backpropagation {adam} (default: adam)
  --adam_b2_range ADAM_B2_RANGE [ADAM_B2_RANGE ...]
                        Adam decay rate for l2 norm moving average (default:
                        '0.9 0.999')
  --annealing_restarts_range ANNEALING_RESTARTS_RANGE [ANNEALING_RESTARTS_RANGE ...]
                        Range of learning rate annealing restarts {0 - off}
                        (default: [2])
  --annealing_factor_range ANNEALING_FACTOR_RANGE [ANNEALING_FACTOR_RANGE ...]
                        Division factor range for annealing learning rate
                        (default: [2])
  --architecture ARCHITECTURE, -a ARCHITECTURE
                        select the deep learning architecture, options =
                        {tsainfcnn} (default: tsainfcnn)
  --learning_rate_range LEARNING_RATE_RANGE [LEARNING_RATE_RANGE ...], -l LEARNING_RATE_RANGE [LEARNING_RATE_RANGE ...]
                        Range for learning rate rage (default: '0.0001 0.001')
  --num_epochs NUM_EPOCHS, -n NUM_EPOCHS
                        Rnage for number of epochs over full training data
                        (default: 50)
  --batch_size_range BATCH_SIZE_RANGE [BATCH_SIZE_RANGE ...], -b BATCH_SIZE_RANGE [BATCH_SIZE_RANGE ...]
                        Size of training and eval batches (default: '50 100
                        150')
  --keep_prob_range KEEP_PROB_RANGE [KEEP_PROB_RANGE ...]
                        Keep probability for dropout (default: '0.7 0.8 0.9')

```

##### Parameters - tsainfcnn

```
  --embedd_filter_sizes_range EMBEDD_FILTER_SIZES_RANGE [EMBEDD_FILTER_SIZES_RANGE ...]
                        Range for filter dimensions of word embedding CNN
                        convolutions (default: '2,3,4,5,6 1,2,3,4,5
                        2,3,3,3,5')
  --n_filters_range N_FILTERS_RANGE [N_FILTERS_RANGE ...]
                        Range for number of convolutions per filter size
                        (default: '100 150')
  --n_dense_output_range N_DENSE_OUTPUT_RANGE [N_DENSE_OUTPUT_RANGE ...]
                        Range for number of units in dense layer (default:
                        '100 150 200 250')
```

##### Command line Example

(see [jasper-local/automated_train_cmd.txt](http://10.188.49.222:3000/jasper/TF-Text-Classification/src/jasper-local/automated_train_cmd.txt) for more examples)

```
nohup \
python automated_train.py \
--max_running_time 60 \
--cross_valid 5 \
--experiment_name smm4h_task2_all_godin_wide_filter_0 \
--architecture tsainfcnn \
--learning_rate_range 0.0001 0.001 \
--num_epochs 100 \
--batch_size_range 50 100 150 \
--document_length 47 \
--embedding_dimension 400 \
--n_classes 3 \
--train_path /datadrive/nlp/jasper/smm4h/npy/smm4h_task2_all_godin_400.npy \
--no_one_hot \
--keep_prob_range 0.4 0.5 0.6 0.7 0.8 0.9 \
--embedd_filter_sizes_range 1,2,3,4,5 2,3,4,5,6 3,4,5,6,7 1,2,2,2,3 2,3,3,3,4 3,4,4,4,5 4,5,5,5,6 \
--n_filters_range 100 200 300 400 \
--n_dense_output_range 100 200 300 400 \
--verbose 1 \
--eval_metric f1_micro \
--eval_labels 0 1 \
>> smm4h_task2_all_godin_wide_filter_0.out
```