"""
basics.py is a library of utility functions meant
to expedite boiler plate actions like loading up data
"""

import sys
import os
import csv

def csv2data(filename, has_header=True):
    """
    read the data in a .csv file with a header
    and return a matrix M of data.

    args:
        --> filename : the .csv file
        --> has_header : defaulted to True; specify whether
                         the input file has a header or not
    returns:
        --> data matrix M: the rows x columns of the .csv,
                         the header of the file
    """
    if os.path.exists(filename):

        if filename.endswith(".csv"):
            M = []
            with open(filename, "r") as csvfile:
                reader = csv.reader(csvfile)

                header = None

                if has_header:
                    header = next(reader, None)

                for row in reader:
                    M.append(row)

                return M, header

        else:
            raise ValueError("%s doesn't end in .csv!" % filename)

    else:
        raise FileNotFoundError("Could not locate %s" % filename)


def check_exists_create_if_not(directory, suppress=False):
    """
    check whether a directory exists -- create it if it doesn't
    
    args:
        directory: name of directory to test

    returns:
        Nothing
    """
    
    if "." in directory:
        raise ValueError("looks like %s isn't a valid directory name" % directory)

    if not os.path.isdir(directory):
        if not suppress:
            print("[check_exists_create_if_not] making a dir: %s" % directory)
        os.makedirs(directory)
