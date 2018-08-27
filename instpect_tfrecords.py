import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import argparse
parser = argparse.ArgumentParser(description='Optional app description')

def main(input, num_rec_to_show=5):
    cnt = 0
    print("####################%d records:################" % num_rec_to_show)
    for example in tf.python_io.tf_record_iterator(input):
        jsonMessage = MessageToJson(tf.train.Example.FromString(example))
        if cnt < num_rec_to_show:
            print(jsonMessage)
        cnt +=1
        # result = tf.train.Example.FromString(example)
        # print(result)

    print("Total number of recirds: %d" %cnt)



if __name__ == "__main__":
    # Required
    parser.add_argument('input', type=str,
                        help='input file')
    # Optional
    parser.add_argument('--numrec', type=int, default=5,
                        help='number of records to show')

    args = parser.parse_args()
    main(args.input, args.numrec)
