#!/usr/bin/env python3

import sys
import csv
import random
from collections import defaultdict
from xml.etree import ElementTree
import numpy as np
from sklearn import preprocessing
"""
Currently supported data formats:
    identifier  options     format
    FT1         -           __label__<label> TAB <text>
    STD1        i           <id (i)> TAB <label> TAB <text>

TODO:
- change file access to with
- parameter documentation
"""

DEFAULT_OUT_FORMAT = "STD1i"
DEFAULT_IN_FORMAT = "STD1i"


def _line_to_format(data_format, data):
    """
    Convert data to standard format by data_format identifier.
    """
    if data_format.startswith("FT1"):
        line = _line_to_FT1(data_format, data)
    elif data_format.startswith("STD1"):
        line = _line_to_STD1(data_format, data)
    else:
        raise ValueError("Invalid data_format: {0:s}".format(data_format))
    return line


def _line_to_FT1(format, data):
    """
    Convert data to FT1 format by data_format identifier.

    Input dictionary data must include values for <label, text> items.
    """
    if "label" not in data:
        raise ValueError("Required item data missing.")
    if "text" not in data:
        raise ValueError("Required item text missing.")

    label = "__label__" + clean_label(data["label"])
    text = clean_text(data["text"])

    line = label + "\t" + text
    return line


def _line_to_STD1(format, data):
    """
    Convert data to STD1 format by data_format identifier.

    Input dictionary data must include values for <label, text, id (STD1i)> items.
    """
    if "label" not in data:
        raise ValueError("Required item data missing.")
    if "text" not in data:
        raise ValueError("Required item text missing.")
    if "i" in format:
        if "id" not in data:
            raise ValueError("Required item id missing.")

    label = clean_label(data["label"])
    text = clean_text(data["text"])
    id = data["id"]

    line = id + "\t" + label + "\t" + text
    return line


def clean_text(text):
    """
    Clean text string.
    
    Replaces newline characters by space and removes carriage return.
    """
    clean_text = text.replace('\n', ' ').replace('\r', '').strip()
    return clean_text
    
    
def clean_label(label):
    """
    Clean label string.
    
    Replaces newline characters by space, removes carriage return and lowercases.
    """
    clean_label = (
        label
        .replace('\n', ' ')
        .replace('\r', '')
        .replace(" ", "_")
        .lower()
        .strip()
        )
    return clean_label


def check_for_data_format(path):
    """
    Check for standard data format identifiers in path.
    
    Data format identifier must be terminated by a '-', '_' or '.' character.
    """
    identifier = None
    if "FT1" in path:
        identifier = _extract_identifier("FT1", path)
    if "STD1" in path:
        identifier = _extract_identifier("STD1", path)

    return identifier
    

def _extract_identifier(identifier, path):
    """
    Extract full identifier from a path.
    
    Data format identifier must be terminated by a '-', '_' or '.' character.
    Closet of these character terminates identifier.
    """
    start_pos = path.find(identifier)
    start_cut = path[start_pos:]
    
    end_pos_under = start_cut.find("_")
    if end_pos_under == - 1:
        end_pos = float("inf")
    else:
        end_pos = end_pos_under
        
    end_pos_dash = start_cut.find("-")
    if end_pos_dash != -1 and end_pos_dash < end_pos:
        end_pos = end_pos_dash
        
    end_pos_dot = start_cut.find(".")
    if end_pos_dot != -1 and end_pos_dot < end_pos:
        end_pos = end_pos_dot
    
    if end_pos == float("inf"):
        raise ValueError("Data format identifier in file path not terminated "
                         "by '-', '_' or '.'.")
    
    full_identifier = start_cut[0:end_pos]
    
    return full_identifier


def _append_data_format(path, data_format):
    """
    Append underscore format string at the end of file name.
    """
    name_end = path.rfind(".")
    new_path= path[:name_end] + "_" + data_format + path[name_end:]
    return new_path


def _check_format(path, data_format):
    """
    Check output path against format. Path has priorty if formats differ.
    
    Ajusts output_format from path if exists or appends output_format to path
    if doesn't exist.
    """
    file_name_data_format = check_for_data_format(path)
    if file_name_data_format is None:
        print("WARN: No valid format identifier found in output path. "
              "Designated format is '" + data_format + "'.")
        path = _append_data_format(path, data_format)
    else:
        if file_name_data_format != data_format:
            print("WARN: Path format '" + file_name_data_format + "' in "
                  "conflict with set format '" + data_format + "'."
                  "Designated format is '" + file_name_data_format + "'.")
            data_format = file_name_data_format
        
    return path, data_format
     
     
def convert_tsv_to_tsv(input, output_path, output_format=DEFAULT_OUT_FORMAT):
    """
    Convert tsv input file into tsv output file.
    
    """
    #Check output path against format. Path has priorty if formats differ.
    output_path, output_format = _check_format(output_path, output_format)
    
    with open(output_path, "w") as fo:
        count = 0
        for input_file in input.split(','):
            with open(input_file, 'r', encoding='utf-8', ) as f:
                reader = csv.reader(f, delimiter='\t', quotechar = None)
                for rows in reader:
                    try:
                        data = _read_format_rows(rows, DEFAULT_IN_FORMAT)
                        fo.write(_line_to_format(output_format, data)+ "\n")
                    except IndexError:
                        print("IndexError at item " + str(count) + ": " + str(rows))
                    count += 1

    print('Converted ' + str(count) + ' elements of ' + input + ' to ' + output_path)
    return None

    
def convert_csv_to_tsv(input, output_path, output_format=DEFAULT_OUT_FORMAT):
    """
    Convert csv input file into tsv output file.
    
    """
    #Check output path against format. Path has priorty if formats differ.
    output_path, output_format = _check_format(output_path, output_format)
        
    fo = open(output_path, "w")
    
    count = 0
    for inputFile in input.split(','):    
        f = open(inputFile, 'r', encoding='utf-8', )
        try:
            reader = csv.reader(f)
            for rows in reader:
                try:
                    data = _read_format_rows(rows, DEFAULT_IN_FORMAT)
                    fo.write(_line_to_format(output_format, data)+ "\n")
                except IndexError:
                    print("IndexError at item " + str(count) + ": " + rows)
                count += 1
        finally:
            f.close()
            
    fo.close()
    print('Converted ' + str(count) + ' elements of ' + input + ' to ' + output_path)
    return None


def convert_custom_tsv_to_tsv(input, output_path, label_col, text_col, id_col=None, skip_header=True,
                              output_format=DEFAULT_OUT_FORMAT):
    """
    Convert custom csv input file into tsv output file.

    Output format is assumed as DEFAULT_OUT_FORMAT.
    """
    convert_custom_input_to_tsv(input, "\t", output_path, label_col, text_col, id_col=id_col, skip_header=skip_header,
                                output_format=output_format)
    return None


def convert_custom_csv_to_tsv(input, output_path, label_col, text_col, id_col=None, skip_header=True,
                              output_format=DEFAULT_OUT_FORMAT):
    """
    Convert custom csv input file into tsv output file.

    Output format is assumed as DEFAULT_OUT_FORMAT.
    """
    convert_custom_input_to_tsv(input, ",", output_path, label_col, text_col, id_col=id_col, skip_header=skip_header,
                                output_format=output_format)
    return None


def convert_custom_input_to_tsv(input, delimiter, output_path, label_col, text_col, id_col=None, skip_header=True,
                                output_format=DEFAULT_OUT_FORMAT):
    """
    Convert custom input file into tsv output file.

    Output format is assumed as DEFAULT_OUT_FORMAT.
    """
    # Check output path against format. Path has priority if formats differ.
    output_path, output_format = _check_format(output_path, output_format)

    fo = open(output_path, "w")

    count = 0
    missing_content = 0
    for inputFile in input.split(','):
        f = open(inputFile, 'r', encoding='utf-8', )
        try:
            reader = csv.reader(f, delimiter=delimiter)
            if skip_header:
                next(reader, None)
            for rows in reader:
                try:
                    data = _read_custom_rows(rows, label_col, text_col, id_col)
                    if data["label"] == "" or data["text"] == "":
                        missing_content += 1
                        continue
                    fo.write(_line_to_format(output_format, data) + "\n")
                except IndexError:
                    print("IndexError at item " + str(count) + ": " + rows)
                count += 1
        finally:
            f.close()

    fo.close()
    print('Missing label or text for ' + str(missing_content) + ' elements of ' + input)
    print('Converted ' + str(count) + ' elements of ' + input + ' to ' + output_path)

    return None

    
def convert_xml_to_tsv(
        input, output_path, label_xpath, text_xpath, output_format=DEFAULT_OUT_FORMAT):
    """
    Convert xml input file into tsv output file.
    
    Input format is assumed as STD1.
    """
    #Check output path against format. Path has priorty if formats differ.
    output_path, output_format = _check_format(output_path, output_format)
    
    fo = open(output_path, "w")

    count = 0
    for inputFile in input.split(','):    
        f = open(inputFile, 'r', encoding='utf-8', )
        try:
            for line in f:
                try:
                    root = ElementTree.fromstring(line)
                    data = {}
                    data["label"] = root.find(label_xpath).text.lower().strip()
                    data["text"] = root.find(text_xpath).text.strip()
                    fo.write(_line_to_format(output_format, data)+ "\n")
                    count += 1
                except ElementTree.ParseError:
                    print(line)
                except AttributeError:
                    print(line)
        finally:
            f.close()
            
    fo.close()
    
    print('Converted ' + str(count) + ' elements of ' + input + ' to ' + output_path)
    return None
    
    
def read_data(file_path, input_format=DEFAULT_IN_FORMAT):
    """
    Read data from input path as specified format.
    """
    csv.field_size_limit(sys.maxsize)

    file_path_input_format = check_for_data_format(file_path)
    if file_path_input_format is None:
        print("WARN: No standard format identifier in file_name. Will use "
            "input format: '" + input_fromat + "'.")
    else:
        input_format = file_path_input_format
        
    f = open(file_path, 'rt')
    labels = []
    content = []
    unique_labels = []
    try:
        reader = csv.reader(f, delimiter='\t')
        for rows in reader:
            data = _read_format_rows(rows, input_format)
            label = data["label"]
            if label not in unique_labels:
                unique_labels.append(label)
            labels.append(label)
            content.append(data["text"])
    finally:
        f.close()
    return labels, content, unique_labels


def read_data_gen(file_path, input_format=DEFAULT_IN_FORMAT):
    """
    Read data from input path as specified format via generator.

    :param file_path: string
        Input path to tsv file
    :param input_format: string
        Identifier of input_format
    :return: generator dict {<string: string}
        Data extracted from input file via input format

    """
    csv.field_size_limit(sys.maxsize)

    file_path_input_format = check_for_data_format(file_path)
    if file_path_input_format is None:
        print("WARN: No standard format identifier in file_name. Will use "
              "input format: '" + input_fromat + "'.")
    else:
        input_format = file_path_input_format

    with open(file_path, "r", encoding='utf-8') as f:
        #reader = csv.reader(f, delimiter='\t')
        for line in f:
            rows = line.split("\t")
            data = _read_format_rows(rows, input_format)
            yield data


def _read_format_line(line, format):
    """
    Read an input line of specified format.
    """
    rows = line.strip().split("\t")
    return _read_format_rows(rows, format)


def _read_format_rows(rows, format):
    """
    Read split rows of an input line of specified format.
    """
    if format.startswith("FT1"):
        data = _read_FT1_rows(rows, format)
    elif format.startswith("STD1"):
        data = _read_STD1_rows(rows, format)
    else:
        raise ValueError("Invalid format identifier: '" + format + "'.")
    return data


def _read_custom_rows(rows, label_col, text_col, id_col=None):
    """
    Read row items, where content positions are defined manually.

    :param rows:
    :param label_col:
    :param text_col:
    :param id_col:
    :return:
    """
    data = {}
    if id_col is not None:
        data["id"] = rows[id_col].strip()
    data["label"] = clean_label(rows[label_col])
    data["text"] = clean_text(rows[text_col])
    return data


def _read_FT1_rows(rows, format):
    """Read FT1 formatted rows."""
    data = {}
    label = clean_label(rows[0]).replace("__label__", "")
    data["label"] = label
    data["text"] = clean_text(rows[1])
    return data


def _read_STD1_rows(rows, format):
    """Read FT1 formatted rows."""
    data = {}
    id_offset = 0
    if "i" in format:
        data["id"] = rows[0].strip()
        id_offset = 1
    data["label"] = clean_label(rows[0 + id_offset])
    data["text"] = clean_text(rows[1 + id_offset])
    return data


def split(input_file, output_file1, output_file2, proportion, shuffle_seed=23):
    """Split input file into two output files according to proportion input.
    
    Proportion should be a values between 0 and 1. Proportion part of input file
    goes into first output file and 1-poportion part goes into second.
    """
    if proportion <= 0 or proportion >= 1:
        raise ValueError("Proportion outside of valid range, ]0.0, 1.0[.")
        
    lines = open(input_file, 'r', encoding='utf-8').readlines()
    n_lines = len(lines)
    random.Random(shuffle_seed).shuffle(lines)
    open(output_file1, 'w', encoding='utf-8').writelines(lines[0:-int(n_lines*proportion)])
    open(output_file2, 'w', encoding='utf-8').writelines(lines[-int(n_lines*proportion):])
    return None
    

def remove_tsv_cols(input_path, output_path, col_ids):
    """
    Remove tsv columns from input path file and save to output path.

    :param input_path: string
        Input path to tsv file
    :param output_path: string
        Output path to tsv files
    :param col_ids: list <int>
        Id/s of column/s to remove
    :return: None
    """

    with open(output_path, "w") as fo:
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=None)
            for in_rows in reader:
                try:
                    out_rows = []
                    for i in range(len(in_rows)):
                        if i in col_ids:
                            continue
                        else:
                            out_rows.append(in_rows[i])

                    fo.write("\t".join(out_rows) + "\n")
                except IndexError:
                    print("IndexError at item " + str(count) + ": " + str(rows))
                count += 1

    print('Removed column/s' + str(col_ids) + ' of ' + str(count) + ' lines of ' + input + ' to ' + output_path)

    return None


def get_n_lines(input_path):
    """
    Count lines in file. Will not count empty lines.

    Prints warning when finding empty line.

    :param input_path: string
        Path to input file
    :return: int
        Number of line in input file
    """
    count = 0
    with open(input_path, "r") as input_file:
        for line in input_file:
            if line.strip() == "":
                print("WARN: Found empty line while counting lines, will not count.")
                continue
            count += 1
    return count


def append(input_path_1, input_path_2, output_path):
    """
    Append all lines of two input files creating a new output file.

    :param input_path_1: string
        Path to input file one (first part of output)
    :param input_path_2: string
        Path to input file two (second part of output)
    :param output_path: string
    :return: None
    """
    with open(output_path, "w", encoding='utf-8') as fo:
        count = 0
        with open(input_path_1, 'r', encoding='utf-8') as f1:
            for line in f1:
                if line.strip() != "":
                    fo.write(line)
                    count += 1
        with open(input_path_2, 'r', encoding='utf-8') as f2:
            for line in f2:
                if line.strip() != "":
                    fo.write(line)
                    count += 1

    print('Wrote ' + str(count) + ' lines of ' + input_path_1 + ' and ' + input_path_2 + ' to ' + output_path)

    return None


def load_data_npy(path, n_labels, one_hot=True):
    """
    Load numpy data matrix where first column is ids, followed by label columns and data.

    n_labels specifics number of label columns for one-hot encoded labels (default)
    Returns data, one-hot labels and ids

    :param path: string
        Input data file path
    :param n_labels: int
        Specifics number of label columns
    :param one_hot: boolean
        Flag whether input data is already one hot encoded (default: True)
    :return: array of floats, array of ints, array of ints
        Data array, one hot label array, id array
    """
    data = np.load(path)
    # 0th column for ids
    ids = data[:, 0]
    if not one_hot:
        # Convert to one hot
        y = one_hot_conversion(y_basic, n_labels)
        # Data columns
        x = data[:, 2:]
    else:
        # Label column
        y = np.array(data[:, 1:n_labels+1], dtype=int)
        # Data columns
        x = data[:, n_labels+1:]
    return x, y, ids


def one_hot_conversion(labels, n_labels=None):
    """
    Convert input labels to one hot labels.

    :param labels: array of ints
        Input array of ints
    :param n_labels: int
        Number of unique labels, determined by labels input if None (default: None)
    :return: array of ints
        One hot label array
    """
    if n_labels is None:
        n_labels = np.unique(labels).shape[0]

    norm_labels = np.asarray(labels) - np.min(labels, axis=0)
    # print(y_norm) -> array([0, 1, 2])
    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    one_hot_labels = np.eye(n_labels, dtype=int)[norm_labels]
    return one_hot_labels


def encode_labels(labels):
    """
    Transforms labels to normalized int labels via sklearn LabelEncoder.

    String labels will be encoded in sorted order, thus strings are converted to integers by their alphabetical order.

    :param str_labels: array of int/float/string
        Input labels
    :return: array of int
        Array of integer labels starting with 0
    """
    le = preprocessing.LabelEncoder()
    norm_labels = le.fit_transform(labels)
    return norm_labels


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == 'tsv':
        input = sys.argv[2]
        output = sys.argv[3]
        convert_tsv_to_tsv(input, output)
    elif mode == 'csv':
        input = sys.argv[2]
        output = sys.argv[3]
        convert_csv_to_tsv(input, output)
    elif mode == 'csv-c':
        input = sys.argv[2]
        output = sys.argv[3]
        label_col = int(sys.argv[4])
        text_col = int(sys.argv[5])
        id_col = None
        if len(sys.argv) >= 5:
            id_col = int(sys.argv[6])
        convert_custom_csv_to_tsv(input, output, label_col, text_col, id_col)
    elif mode == 'tsv-c':
        input = sys.argv[2]
        output = sys.argv[3]
        label_col = int(sys.argv[4])
        text_col = int(sys.argv[5])
        id_col = None
        if len(sys.argv) >= 5:
            id_col = int(sys.argv[6])
        convert_custom_tsv_to_tsv(input, output, label_col, text_col, id_col)
    elif mode == 'xml':
        input = sys.argv[2]
        output = sys.argv[3]
        label_xpath = sys.argv[4]
        content_xpath = sys.argv[5]
        convert_xml_to_tsv(input, output, label_xpath, content_xpath)
    elif mode == "split":
        split(sys.argv[2], sys.argv[3],  sys.argv[4], float(sys.argv[5]))
    elif mode == "rm_col":
        input = sys.argv[2]
        output = sys.argv[3]
        col_ids = list(map(int, sys.argv[4].split(",")))
        remove_tsv_cols(input, output, col_ids)
    elif mode == "append":
        input1 = sys.argv[2]
        input2 = sys.argv[3]
        output = sys.argv[4]
        append(input1, input2, output)
    else:
        print("Unrecognized mode: {0:s}".format(mode))
