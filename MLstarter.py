# FILE HANDLING
import os
import pathlib as pt

# DATA WRANGLING and MATH
import pandas as pd
import numpy as np
from numba import jit

# GRAPHING
from matplotlib import pyplot as plt
import seaborn as sns
import altair as alt
import plotly.figure_factory as ff
import streamlit as st

context = os.getcwd()
st.write('Context:{}'.format(context))

# EDA
df1 = pd.read_csv(context+'/data/train/train.csv')
if st.checkbox('Show raw data'):
    st.subheader('Our Training Data')
    st.write(df1)

st.subheader('PLOTS')

@st.cache
@jit
def genhist(df, bins):
    hist_values = np.histogram(df,
                               bins=bins,
                               range=(0,bins))[0]
    return hist_values

df1 = df1.fillna(method='pad')

st.subheader('Filter on Column')
datacol = st.radio("SELECT LOG/Column TO PLOT",df1.columns)
plotdata = df1[[datacol]]
sns.set_style("dark")
sns.scatterplot(data=plotdata)
st.pyplot()
st.subheader('HISTOGRAM')
bins = st.slider('bins', 3, 25, 5)
st.bar_chart(genhist(plotdata,bins))


# HMM

if __name__ == '__main__':
    #st.run()
    pass
