from __future__ import print_function
from os import path
import numpy as np
import argparse


def main(in_files, out_dir):

    for npy_path in in_files:
        in_filename = path.basename(npy_path)
        out_filename = in_filename.replace('.npy', '.csv')
        csv_path = path.join(out_dir, out_filename)

        arr = np.load(npy_path)
        np.savetxt(csv_path, arr, delimiter=",")
        print('wrote {}'.format(csv_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs='+')
    parser.add_argument("out_dir")
    args = parser.parse_args()
    main(args.in_files, args.out_dir)
