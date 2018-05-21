#!/bin/sh

#this script will update radiation-benchmark-parser at
#radiation-benchmark repository
#if you do not execute it after push at radiation-benchmark-parser
#radiation-benchmarks repository will be with wrong link

git pull
cd radiation-benchmarks-parsers/
git checkout master && git pull
cd ..
#git submodule update --init --recursive
git add radiation-benchmarks-parsers
git commit -m "updating submodule to latest"
git push -u origin master
