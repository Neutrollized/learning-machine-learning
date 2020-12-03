#!/usr/bin/env python3

import pandas as pd
import numpy as np

# character encoding module
import chardet

np.random.seed(0)


#--------------------------------------

before = "This is the euro symbol: €"
print(before)
print(type(before))

utf8 = before.encode("utf-8", errors="replace")
print(utf8)
print(type(utf8))

ascii = before.encode("ascii", errors="replace")
print(ascii)
print(type(ascii))


#-----------------------------
# Kickstarter example
#-----------------------------
# reading this file will give you errors with 
#kickstarter2016 = pd.read_csv("../data/ks-projects-201612.csv.zip")

# NOTE: here, your data file needs to be unzipped ('open' is not as smart as 'read_csv')
with open("../data/ks-projects-201612.csv", 'rb') as rawdata:
  result = chardet.detect(rawdata.read(10000))

print(result)
#print(kickstarter2016.columns)
