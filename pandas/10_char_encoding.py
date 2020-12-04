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

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
ascii = before.encode("ascii", errors="replace")
print(ascii)
print(type(ascii))

# you can go backwards with decode (c'mon, of course there's a decode!)
new_before = utf8.decode("utf-8")
print(new_before)


#-----------------------------
# Kickstarter example
#-----------------------------
# reading this file will give you errors with 
#kickstarter2016 = pd.read_csv("../data/ks-projects-201612.csv")

# NOTE: here, your data file needs to be unzipped ('open' is not as smart as 'read_csv')
# you may also want to play around with the number of bytes you read as 10k bytes might give you the wrong answer
with open("../data/ks-projects-201612.csv", 'rb') as rawdata:
  result = chardet.detect(rawdata.read(10000))

print(result)


kickstarter2016 = pd.read_csv("../data/ks-projects-201612.csv", encoding='Windows-1252')
print(kickstarter2016.columns)

# you can then opt to write it back out in a more standard UTF-8 format for easier future use
#kickstarter2016.to_csv("ks-projects-201612-utf8.csv")
