import tensorflow as tf
from google.protobuf.json_format import MessageToJson

import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser(description='Optional app description')


def main(inputs, num_rec_to_show=5):
    need_to_be_read_percentage_neg = []
    need_to_be_read_percentage_pos = []
    need_to_be_read_percentage= []
    print("####################%d records:################" % num_rec_to_show)
    inputs = tf.gfile.Glob(os.path.expanduser(inputs))
    print('number of all tf.record files: %d' %len(inputs))
    cnt = 0
    for input in inputs:
        for example in tf.python_io.tf_record_iterator(input):
            jsonMessage = MessageToJson(tf.train.Example.FromString(example))
            if cnt < num_rec_to_show:
                    print(jsonMessage)
            cnt +=1
            # result = tf.train.Example.FromString(example)
            # print(result)

            jsonMessage = json.loads(jsonMessage)
            feature = jsonMessage["features"]["feature"]
            targets = feature["targets"]["int64List"]["value"][0]
            percentage_read = float(feature["percentage_read"]["floatList"]["value"][0])

            if targets == '1':
                need_to_be_read_percentage_pos.append(percentage_read)
            else:
                need_to_be_read_percentage_neg.append(percentage_read)

            need_to_be_read_percentage.append(percentage_read)

    avg_need_to_be_read_percentage = np.mean(np.array(need_to_be_read_percentage))
    avg_need_to_be_read_percentage_pos = np.mean(np.array(need_to_be_read_percentage_pos))
    avg_need_to_be_read_percentage_neg = np.mean(np.array(need_to_be_read_percentage_neg))

    std_need_to_be_read_percentage = np.std(np.array(need_to_be_read_percentage))
    std_need_to_be_read_percentage_pos = np.std(np.array(need_to_be_read_percentage_pos))
    std_need_to_be_read_percentage_neg = np.std(np.array(need_to_be_read_percentage_neg))

    print("#################### STAT :################")
    print("Total number of records: %d" %len(need_to_be_read_percentage))
    print("Total number of positive records: %d" % len(need_to_be_read_percentage_pos))
    print("Total number of negative records: %d" % len(need_to_be_read_percentage_neg))

    print("Average percentage of tokens needed to be read: %f" % avg_need_to_be_read_percentage)
    print("Average percentage of tokens needed to be read in positive examples: %f" % avg_need_to_be_read_percentage_pos)
    print("Average percentage of tokens needed to be read in negative examples: %f" % avg_need_to_be_read_percentage_neg)

    print("STD percentage of tokens needed to be read: %f" % std_need_to_be_read_percentage)
    print("STD percentage of tokens needed to be read in positive examples: %f" % std_need_to_be_read_percentage_pos)
    print("STD percentage of tokens needed to be read in negative examples: %f" % std_need_to_be_read_percentage_neg)


if __name__ == "__main__":
    # Required
    parser.add_argument('input', type=str, help='input file')

    # Optional
    parser.add_argument('--numrec', type=int, default=1,help='number of records to show')

    args = parser.parse_args()
    main(args.input, args.numrec)
