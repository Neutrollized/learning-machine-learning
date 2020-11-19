# README
This directory contains some basics of [pandas](https://pandas.pydata.org/) for data manipulation/cleaning.

## Gotchas!
There's a weird gotcha with `iloc` and `loc` in pandas...

If, for example, you wanted the first 100 rows of some data frame, if you were to use **`iloc`**, the index would be [0:100] (which starts at 0 and ends just before 100, i.e. 99) and is just like how indexes in python works.

Now, if you were to do the same using **`loc`**, specifying [0:100] would actually net you 101 results as it includes the last index value.  In other words, if you wanted to fetch the first 100 rows of something, you need to use [0:99] with `iloc`

## NumPy Random Seed
[Explained](https://www.sharpsightlabs.com/blog/numpy-random-seed/)
