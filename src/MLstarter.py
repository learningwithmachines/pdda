# FILE HANDLING
import os
import pathlib as pt

# DATA WRANGLING and MATRIX MATH
import pandas as pd
import numpy as np

# GRAPHING
from matplotlib import pyplot as plt
import seaborn as sns
# import altair
import streamlit as st

st.write('Context:{}'.format(os.getcwd()))

# EDA
df1 = pd.read_csv('train.csv')
df1.shape

# HMM

if __name__=='__main__':
    #st.run()
    pass
