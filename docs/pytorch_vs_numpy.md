# PyTorch as a Linear-Algebra support package

Pytorch provides utilities to manipulate numbers with linear-algebra style logic and operations.
A data point with two values, for example *age* and *height*, can be represented by a vector with two items, where the first entry is the age, and the second entry is the height.

``` py
import torch


# first entry age in years, second entry height in centimeters
person_details = torch.tensor([18.0, 172.4])

person_details
```

```tensor([ 18.0000, 172.4000])```

Why not a *dataclass* (or just a Python *class*)? Why not a Python *dict*?  
It is still a good practice to use more structured data types in your program and databases. At one stage to communicate the data to machine learning; linear models, neural networks, etc., we shall need probably to work with this kind of representation, meaning vectors, matrices, and higher dimensions tensors, with all entries of the same (numeric) type.

If you are familiar with *DataFrames* from *Pandas*, *Polars*, or *Spark*, or if you have already used tabular data with ML libraries such as *Scikit-Learn*, this seems to be a step backwards.
Yet think of streams of images from cameras (a video or so), embedding for variable-length texts, sound, the feed from sensors, etc. For more flexibility with the inputs and the outputs, getting used to tensor representation will help us with addressing those challenges. We'll need to find the right combination of "human facing" representation and "neural network facing" representation, and where to do the back and forward translation.    

So a vector is tensor with one dimention. To continue with above example, we can have the details for more than one person, in a matrix. Each row will represent another person (another data point.).

``` py
..

# first entry age in years, second entry height in centimeters
subjects_details = torch.tensor([
    [18.0, 172.4],
    [51.0, 169.2],
    [46.0, 164.5],
])

subjects_details
```

```
tensor([[ 18.0000, 172.4000],
        [ 51.0000, 169.2000],
        [ 46.0000, 164.5000]])
```

If you are saying at this stage, "hey, this is just like *NumPy*", then you are right. Tensors in PyTorch are like arrays in NumPy. The functionality of NumPy is mostly available also with PyTorch, and even the inteface of PyTorch was inspired by NumPy and is very similar. So why yet another implementation?

PyTorch's tensors bring additional functionalities that are not found in NumPy. Trying to add the functionalities, we'll describe those later, to NumPy, would have slowed the process and *Meta*, the developers behind PyTorch wanted to run fast. Also NumPy have some target audience and relevant optimizations, that are different from what the creators of PyTorch needed. PyTorch is optimized towards the usage with neural networks. NumPy is more memory optimized.

There are still many ideas in NumPy worth getting to know, that are also relevant while using PyTorch. For example the concept of *vectorization*, which is, instead of looping over the entries and applying an operation to each entry at a time, issue the operation on the whole set of entries at once, where possible, letting the implementation by the package use hardware and other tricks to apply "at the same time" to all entries, or to batches of inputs, transparently to the caller (ex. instead of applying 1,000 times applying 10 times for each batch/slice of size 100). 

An example of things you can do with PyTorch (and with Numpy) is the following: let's say we want to calculate the BMI ( Body Mass Index) for people, based on the following formula:

$$
BMI = {mass(kg) \over height^2(m)}
$$

``` py
..

# first entry age in years, second entry height in centimeters,
# and just added the mass in kilograms.
subjects_details = torch.tensor([
    [18.0, 172.4, 62.1],
    [51.0, 169.2, 66.0],
    [46.0, 164.5, 72.0],
])

subjects_BMI = (
    subjects_details[:, 2] / (subjects_details[:, 1] / 100.0) ** 2
)

subjects_BMI
```

```tensor([20.8938, 23.0538, 26.6073])```

What we did there? We picked the third column (index 2) and devided it (element wise) with the second column (index 1), after the last was first converted to meters, and then raised to the power of 2.  
If I read the results correct, then the third person is a bit overweight.

A side note. Is NumPy obsolete? Can we forget about NumPy, or never bother to learn it if did not play with it yet?
My answer will be that at the moment it is still around and is well supported and integrated by and with many packages. Also I believe it is a dependancy of PyTorch (probably not making use of NumPy, yet as of back-and-forward conversions.). It is very easy to initialize a PyTorch tensor from a NumPy array and vice versa. So keep learning PyTorch, and use NumPy if and where needed.

