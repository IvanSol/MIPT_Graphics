#!/bin/bash
for f in $(cat dependencies.txt)
do
  echo Installing $f
  sudo pip install $f
done
