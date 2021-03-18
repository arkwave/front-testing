# PortfolioSimulator
General-purpose module that simulates the behavior of a portfolio subject to a variety of constraints and strategies.

## Overview:
This module allows arbitrarily complex portfolios to be run through historical price/vol series of variable length. The position is run through the historic data, and the outputs (cumulative PnL, gamma PnL, vega PnL, daily PnL and future prices) are plotted using Plotly.  

## Requirements:
1. Python > 3.x
2. Packages: Pandas, Numpy, Scipy for most of the heavy lifting. Plotly for visualization. Using the Anaconda distribution recommended.
