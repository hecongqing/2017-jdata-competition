#!/bin/bash
python preprocessing.py
python USModel.py
python Umodel_0.py
python Umodel_1.py
python Umodel_2.py
python merge_result.py

