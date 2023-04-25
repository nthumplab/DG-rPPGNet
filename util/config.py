# parameter setting
import argparse


def get_args():
    
    parser = argparse.ArgumentParser()
    
    # ----------------- General ------------------
    parser.add_argument('--train_dataset', default="", type=str,
                        help="""
                        Options => C: COHFACE, P: PURE, U: UBFC
                        e.g. --dataset="C"  for intra-training/testing on COHFACE
                             --dataset="UP" for cross-training/testing on PURE and UBFC        
                        """)
    parser.add_argument('--test_dataset', default="", type=str,
                        help="Same as above")
    
    parser.add_argument('--in_ch', default=3, type=float, 
                        help="input channel, you may change to 1 if dataset type is NIR")

    parser.add_argument('--model_S', default=2, type=int,
                        help="spatial dimension of model")
    parser.add_argument('--conv', default="conv3d", type=str,
                        help="Convolution type for 3DCNN")
        
    parser.add_argument('--bs', default=1, type=int,
                        help="batch size")
    parser.add_argument('--epoch', default=400, type=int,
                        help="training/testing epoch")
    parser.add_argument('--fps', default=30, type=int,
                        help="fps for dataset")
    parser.add_argument('--lr', default=2e-4, type=float,
                        help="learning rate")
    
    parser.add_argument('--high_pass', default=40, type=int)
    parser.add_argument('--low_pass', default=250, type=int)
    
    # ----------------- Training -----------------
    parser.add_argument('--train_T', default=2, type=int,
                        help="training clip length(seconds))")
    
    parser.add_argument('--fine_tune', default=False, type=bool,
                        help="Is fine-tune")
    
    # ----------------- Testing -----------------
    parser.add_argument('--test_T', default=7, type=int,
                        help="testing clip length(seconds))")
    
    parser.add_argument('--inject_noise', action='store_true')
    
    return parser.parse_args()


def get_name(args, train=True):
    
    trainName = f"{args.train_dataset}_{args.conv}_train_T{args.train_T}"
    testName  = f"{args.train_dataset}_to_{args.test_dataset}_{args.conv}_test_T{args.test_T}"
    
    if args.inject_noise:
        testName += "_injectNoise"
        
    return trainName, testName