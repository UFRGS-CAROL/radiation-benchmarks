"""
Common parameters and functions
"""

import os


class Codes:
    SUCCESS = 0
    ERROR = 1
    CTRL_C = 130


def execute_command(cmd):
    tmp_file = "/tmp/server_error_execute_command"
    result = os.system(f"{cmd} 2>{tmp_file}")
    with open(tmp_file) as err:
        if len(err.readlines()) != 0 or result != 0:
            return Codes.ERROR
    return Codes.SUCCESS
