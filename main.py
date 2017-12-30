import argparse as arg
import emotion_detector as em

parser = arg.ArgumentParser(description='Simple emotion detector.')

parser.add_argument('-n', default=100, type=int, help='The number of data')
parser.add_argument('-b', default=32,  type=int, help='Batch size of dataset')
parser.add_argument('-e', default=100, type=int,  help='Batch size of dataset')
parser.add_argument('-v', default=10,  type=int, help='Interval epoch of lerning progress')
parser.add_argument('-r', default="0.01", type=float, help='learning rate')
parser.add_argument('-d', default="", help='Dump dataset and the learning model as pickle')
parser.add_argument('-l', default="", help='Load dataset and the learning model as pickle')
parser.add_argument('-t', action='store_true', default=False, help='If this option is added, dataset will be trained')
parser.add_argument('-a', action='store_true', default=False, help='If this option is added, dataset contains your face is aquired using web camera')

args = parser.parse_args()


print(args.n)
a = em.emotion_detector(num_datasets=args.n)
if len(args.l):
    a = a.load_pickle(args.l)
    print("pickle data loaded.")

if args.a:
    a.get_train_data("default", False)

if len(args.d):
    a.dump_pickle(args.d)
    print("pickle data dumped.")


#a = a.load_pickle()
if args.t:
    print("train start.")
    a.train(lr=args.r, n_batch=args.b, n_epoch=args.e, n_view=args.v)

if len(args.d):
    a.dump_pickle(args.d)
    print("pickle data dumped.")
#a.detect()
a.detect()


