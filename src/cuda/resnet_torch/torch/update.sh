#!/usr/bin/env bash

git fetch
git reset --hard
# Submodule update is done inside install.sh
./install.sh -s
