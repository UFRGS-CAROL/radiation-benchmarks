from enum import Enum


class ErrorCodes(Enum):
    """
    Error codes used to identify status in this test
    """
    SUCCESS = 0

    # Codes for Machine class
    WAITING_BOOT_PROBLEM = 1
    REBOOTING = 2
    BOOT_PROBLEM = 3
    WAITING_FOR_POSSIBLE_BOOT = 4
    MAX_SEQ_REBOOT_REACHED = 5
    TURN_ON = 6

    # Codes for RebootMachine
    GENERAL_ERROR = 7
    HTTP_ERROR = 8
    CONNECTION_ERROR = 9
    TIMEOUT_ERROR = 10

    def __str__(self):
        """
        Override the str method
        :return:
        """
        return self.name
