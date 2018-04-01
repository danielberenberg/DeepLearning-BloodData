"""
header script for train/testing models.
"""

import argparse
import sys
import os
import basic_utils.basics as base


def parse_input():
    """
    parse the input to the script
    
    args:
        None

    returns:
        parser : argparse.ArgumentParser - the namespace containing 
                 all of the relevant arguments
    """

    parser = argparse.ArgumentParse(description="a suite of commands for running a model")

    parser.add_argument("model_type",
                        help="the type of model to run",
                        type=str,
                        choices=["CNN+LSTM", "3D-CNN"])

    parser.add_argument("--train",
                        help="states whether the model should be trained",
                        type=bool,
                        default=False,
                        action="store_true")

    parser.add_argument("--test",
                        help="states whether the model should be tested",
                        type=bool,
                        default=False,
                        action="store_true")

    parser.add_argument("--batch_size",
                        help="size of batch",
                        type=int,
                        default=4)

    parser.add_argument("--epochs",
                        help="if training, the number of epochs to train for",
                        type=int,
                        default=100)

    parser.add_argument("--output_dir",
                        help="the output directory",
                        type=str,
                        default="outputs")
    
    parser.add_argument("--input_dir",
                        help="the input directory when testing model",
                        type=str,
                        default=None)
    return parser


class ArgumentError(Exception):
    """
    custom exception to thrown due to bad parameter input
    """
    pass


def generate_output_directory(output_dirname):
    """
    create an output directory that contains looks like this:
    
    output_dirname/
        models/
    
    args:
        output_dirname : string - the ouput directory name
    
    returns:
        output_dir : string - name of output_directory
    """

    model_dir = os.path.join(output_dirname, "models")

    base.check_exists_create_if_not(output_dirname)
    base.check_exists_create_if_not(model_dir)

    return output_dirname


def verify_directory_structure(dirname):
    """
    verify that the directory provided conforms to a structure
    that looks like this:

    dir_name/
        models/
        train.csv
        validation.csv
        test.csv
    
    args:
        dirname : string - directory in question

    returns:
        verified : bool - whether or not this directory structure
                          is satisfactory 
    """
    
    verified = True

    if os.path.isdir(dirname):
        if not os.path.isdir(os.path.join(dirname, "models")):
            print("[verify_directory_structure] - no %s/models/ directory" % dirname)
            verified = False
        
        if not os.path.exists(os.path.join(dirname, "train.csv")):
            print("[verify_directory_structure] - no %s/train.csv" % dirname)
            verified = False

        if not os.path.exists(os.path.join(dirname, "validation.csv")):
            print("[verify_directory_structure] - no %s/validation.csv" % dirname)
            verified = False

        if not os.path.exists(os.path.join(dirname, "test.csv")):
            print("[verify_directory_structure] - no %s/test.csv" % dirname)
            verified = False

        return verified

    else:
        return False
    

def validate_arguments(args):
    """
    validate the arguments provided, handle bad/incomplete input

    args:
        args - the arguments provided to the script throught parse_input

    returns:
        None
    """
    
    # if --test=False and --train=False, exit because there's nothing to do
    if (not args.train) and (not args.test):
        raise ArgumentError("Both --train and --test were provided as False " +
                            "exiting because there's nothing to do ...")

    # if batch_size was provided to be negative, exit for bad input
    if args.batch_size < 0:
        raise ArgumentError("The --batch_size should be > 0; " +
                            "got %d" % args.batch_size)

    # if epochs was provided to be negative, exit for bad input
    if args.batch_size < 0:
        raise ArgumentError("The --epochs should be > 0; " +
                            "got %d" % args.epochs)
    
    # if --test was provided only
    if args.test and not args.train:
        # if no input directory specified, exit for bad input
        if args.input_dir is None:
            raise ArgumentError("--test was specified but found no input " +
                                "directory, provide an input directory to " +
                                "test a model only")
        
        # an input directory was specified but if doesn't exist
        if not os.path.isdir(args.input_dir):
            raise ArgumentError("Cannot find input directory %s" % args.input_dir)

        # verify input directory structure
        if not verify_directory_structure(args.input_dir):
            raise ArgumentError("Problems with directory structure of %s" % args.input_dir)
         
    # if --test=True and --train=True, then we need only an output directory
    if args.train and args.test:
        generate_output_directory(args.output_dir)
        args.input_dir = args.output_dir


if __name__ == "__main__":
    sys.exit("under construction ... ")
