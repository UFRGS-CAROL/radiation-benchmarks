from enum import Enum


class ErrorCodes(Enum):
    """
    Error codes used to identify status in this test
    """
    SUCCESS = 0

    # Codes for Machine class
    WAITING = 1
    REBOOTING = 2
    BOOT_PROBLEM = 3
    MAX_SEQ_REBOOT_REACHED = 4
    TURN_ON = 5

    # Codes for RebootMachine
    GENERAL_ERROR = 6
    HTTP_ERROR = 7
    CONNECTION_ERROR = 8
    TIMEOUT_ERROR = 9
