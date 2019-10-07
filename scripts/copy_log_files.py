#!/usr/bin/python3

import argparse
import remote_sock_parameters as par
import datetime

from paramiko import SSHClient
from scp import SCPClient

def scpAllFilesFromDevice():

    pass


def tarToTimeFolder():
    """
    Tar to time
    :return:
    """
    pass



def main(outFolder):
    """
    Main function
    :return: void
    """
    print(outFolder)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy files and generate tar files.')

    parser.add_argument('--out_folder', dest='outFolder', default='./', help='Output folder to save the tar files')

    args = parser.parse_args()

    main(outFolder=args.outFolder)

