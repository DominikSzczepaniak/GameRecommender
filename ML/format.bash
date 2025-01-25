#!/bin/bash

isort ./lightFMscaNN
black ./lightFMscaNN
yapf --in-place --recursive ./lightFMscaNN

isort ./cleora
black ./cleora
yapf --in-place --recursive ./cleora

isort ./funksvd
black ./funksvd
yapf --in-place --recursive ./funksvd

isort ./knn
black ./knn
yapf --in-place --recursive ./knn

isort ./widendeep
black ./widendeep
yapf --in-place --recursive ./widendeep

isort ./popular
black ./popular
yapf --in-place --recursive ./popular

isort ./metrics.py
black ./metrics.py
yapf --in-place --recursive ./metrics.py

isort ./random
black ./random
yapf --in-place --recursive ./random
