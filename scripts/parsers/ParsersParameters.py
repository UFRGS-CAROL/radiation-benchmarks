"""check which parser is going to execute
"""
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Parse logs for Neural Networks')
    parser.add_argument('--get_info', dest='toGetInfo', help='If you want to retrieve all #Info',
                        default='no_info', type=str,required=True)

    args = parser.parse_args()

    return args




