import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print(sys.version_info)

for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__, module.__version__)
print(tf.__path__)

