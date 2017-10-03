#!/bin/bash

# Open file descriptor (fd) 3 for read/write on a text file.
truncate -s 0 /etc/lightdm/lightdm.conf
exec 3<> /etc/lightdm/lightdm.conf

    # Let's print some text to fd 3
    echo "[SeatDefaults]" >&3
    echo "autologin-user=" >&3
# Close fd 3
exec 3>&-
reboot
