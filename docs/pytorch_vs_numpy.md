# PyTorch as a Linear-Algebra support package

Pytorch provides utilities to manipulate numbers with linear-algebra style logic and operations.
A data point with two values, for example *age* and *height*, can be represented by a vector with two items, where the first entry is the age, and the second entry is the height.

``` py
import torch


# first entry age in years, second entry height in centimeters
person_details = torch.tensor([18, 172.4])

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
    [18, 172.4],
    [51, 169.2],
    [46, 164.5],
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
    [18, 172.4, 62.1],
    [51, 169.2, 66.0],
    [46, 164.5, 72.0],
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

It is a good practice to verify the type of a Python object, when in doubt.

``` py
type(subjects_details)
```

```torch.Tensor```

To get a bit more information with tensors, use also the following:

``` py
subjects_details.type()
```

```'torch.FloatTensor'```

``` py
subjects_details.dtype
```

```torch.float32``` (the elements are all of torch.float32 type)

``` py
subjects_details.shape
```

```torch.Size([3, 3])``` (3 rows and 3 columns right after we've added also the mass)

Tensors are very useful also to represent images.

``` py
..
img = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])

img
```

```
tensor([[0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]])
```

``` py
img.type()
```

```'torch.LongTensor'```

Above the implementation of ```torch.tensor``` noted that all the entries are integers and so we got a ```LongTensor```.  
If you want to have floats in the first place, use ```tensor.Tensor``` instead:

``` py
img = torch.Tensor([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])

img
```

```
tensor([[0., 1., 1., 0.],
        [1., 0., 0., 1.],
        [1., 0., 0., 1.],
        [0., 1., 1., 0.]])
```

Alternatively you can convert an existing tensor to another type, by a match operation, or explicitly using ```to```:

```py
img = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])

img.to(torch.float32)
```

```
tensor([[0., 1., 1., 0.],
        [1., 0., 0., 1.],
        [1., 0., 0., 1.],
        [0., 1., 1., 0.]])
```

Images often have multiple channels (ex. RGB). With PyTorch the channel should often come before the height and the width, so if you happen to have HWC and you want to "change" it into CHW, you can do the following:

``` py
img = ... # a tensor representing an image, given in HWC format.
img = img.permute([2, 0, 1]) # let's have it in CHW format.
```

To verify above snippet of code, I've ended up creating first CHW image and then converting it to HWC (just to convert it back):

``` py
chw_img = img.unsqueeze(0).repeat(3, 1, 1)
chw_img
```

*unsqeeuze* adds a dimention. *repeat* in this example "inflates" it from B/W to RGB (three channels).

```
tensor([[[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],

        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],

        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]])
```

``` py
chw_img[1] *= 2
chw_img[2] *= 3
chw_img
```

```
tensor([[[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],

        [[0, 2, 2, 0],
         [2, 0, 0, 2],
         [2, 0, 0, 2],
         [0, 2, 2, 0]],

        [[0, 3, 3, 0],
         [3, 0, 0, 3],
         [3, 0, 0, 3],
         [0, 3, 3, 0]]])
```

Lastly let's show what we wanted, so if the channel is the last dimension HWC:

``` py
hwc_img = chw_img.permute(1, 2, 0)
hwc_img[:, :, 2]
```

```
tensor([[0, 3, 3, 0],
        [3, 0, 0, 3],
        [3, 0, 0, 3],
        [0, 3, 3, 0]])
```

Above demonstrates what the third channel (Blue?) looks like.
Now we can change the channel to be in the first dimension (as usually is assumed with PyTorch).

``` py
hwc_img.permute(2, 0, 1)
```

```
tensor([[[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],

        [[0, 2, 2, 0],
         [2, 0, 0, 2],
         [2, 0, 0, 2],
         [0, 2, 2, 0]],

        [[0, 3, 3, 0],
         [3, 0, 0, 3],
         [3, 0, 0, 3],
         [0, 3, 3, 0]]])
```

It is usually a good practice to operate on tensors and get new tensors as a result. This is more "functional" in the sense that code and functions that we run do not have side effects on the inputs, and shall return the same results (subject to implicit randomness), if called again with the same inputs.
If we do need to save space, we can opt in to "in place" operations.

A very useful functionality of tensor is *reshape*. Imagine a 2d image. Its pixels' values may be stored on disk in a sequential mannner. When we load the image, we are starting with a 1d-tensor. Then maybe we want to process the image with its "georgaphy" in mind (cropping for example).

``` py
img_vec = torch.arange(9)
print(img_vec.shape)
img_mat = img_vec.reshape(-1, 3)
print(img_mat.shape)
```

```
torch.Size([9])
torch.Size([3, 3])
```

Above we've suggested that there are 3 columns, and let PyTorch do the math regarding the rows (this was the -1).

``` py
img_mat
```

```
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
```

When makes sense, PyTorch attempts to avoid moving memory from place to place, but rather only keep "management" information regarding the shape. This however means that there is only one "place":

``` py
img_mat[-1, -1] = 88
img_vec
```

```
tensor([ 0,  1,  2,  3,  4,  5,  6,  7, 88])
```

We've seen above element wise operations, for example when we've computed BMI. PyTorch, as does NumPy, supports also linear algebra style vector and higher-order tensors operations, such as multiplication.

``` py
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
res = torch.matmul(tensor1, tensor2)
print(res)
print(res.size())
print(res.item())
```

```
tensor(-1.0853)
torch.Size([])
-1.0853235721588135
```

We got above a 0-dimensions tensor (a scalar). In order to take the single value out of the tensor,
we've used *.item()*.

## Broadcasting

When we need element-wise operations, such as addition, the tensors are supposed to be of the same size, that is same dimensios, and same number of elements at each dimension. But we've already seen a division by scalar, which means that every element is diveded by the same scalar, or you can also think of it that a conceptual tensor is created, and the scalar is broadcasts to all elements of the new tensor, and then we have the element-wise division. Broadcasting is even more general. If you have two tensors, with different sizes, it can still work, if it is possible to "understand" what is meant; starting from the last dimension, if it is equal among the tensors, we move to the next one. If one of the dimentions is 1, then we assume all entries of the conceptual tensor are the same. If we ran out of dimension in one tensor, we can assume we have there 1 as above.

Here are some examples that should work:

```
[3, 4, 5] and [3, 1, 5]
[1, 2, 2] and [2, 2, 2]
[2, 2] and [2, 2, 2]
```

And here are some examples that should probably don't work:

```
[3, 3] and [3, 6]
[2, 2] and [3, 2]
```

If we did not have the support of broadcasting, we could have just do as above with *unsqueeze*, *squeeze*, *repeat*, etc. However it makes our life a bit easier, and potentially saves memory and computations.

``` py
img = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])

img * torch.tensor([[[1]], [[2]], [[3]]]) # [4, 4] and [3, 1, 1]
```

```
tensor([[[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],

        [[0, 2, 2, 0],
         [2, 0, 0, 2],
         [2, 0, 0, 2],
         [0, 2, 2, 0]],

        [[0, 3, 3, 0],
         [3, 0, 0, 3],
         [3, 0, 0, 3],
         [0, 3, 3, 0]]])
```

## Support for GPU

Tensors can be allocated on the GPU's memory, if you have such. Relevant operations on the tensors will run there (on the GPU), often faster. Operations between tensors need to be done when both tensors are located on the same *device*. If you detect that you have a GPU, and you train a model, putting the parameters of the model on the GPU, the training and evaluation data, should also be at the time on the GPU. Data is often loaded from the disk into the main memory, and then mini-batches are copied to the GPU's memory, used for an iteration, and make space for the next mini-batches, that are waiting either in memory or still on the disk. NumPy as far as I know does not have support for GPU. There is another NumPy like package called Numba, that apparently does support GPU. I haven't played with Numba yet.

``` py
torch.cuda.is_available()
```

```
True
```

Above was shown on [Colab](https://colab.research.google.com/). If like me you don't have GPU in your private environment, consider using Colab.

``` py
tensor1 = torch.randn(3)
tensor1.device
```

```device(type='cpu')```

``` py
tensor1_cuda = tensor1.to("cuda")
tensor1_cuda.device
```

```device(type='cuda', index=0)```

Just to iterate, operations between tensors that are located on different devices, are not supposed to work:

``` py
...

tensor1 = torch.randn(3)
tensor2 = torch.randn(3).to("cuda")
tensor1 + tensor2
```

```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-2-5076b2d4339c> in <cell line: 3>()
      1 tensor1 = torch.randn(3)
      2 tensor2 = torch.randn(3).to("cuda")
----> 3 tensor1 + tensor2

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

If you do want to add as an example two tensors that are on different device, you need first to move one of them to the same device where the other is. Remember that the GPU is fast, while the CPU (the relevant memory) is potentially bigger, but also when needed, backed by even larger space transparently (the operating system's virtual memory and the disk). 

There is more to PyTorch of course, and also related to tensors. We'll talk about *auto grad* next.
