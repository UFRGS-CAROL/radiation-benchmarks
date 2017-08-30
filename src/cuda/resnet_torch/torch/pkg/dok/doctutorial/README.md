<a name="torch.tutorial"/>
# Torch Tutorial #

So you are wondering how to work with Torch?
This is a little tutorial that should help get you started.

By the end of this tutorial, you should have managed to install torch
on your machine, and have a good understanding of how to manipulate
vectors, matrices and tensors and how to build and train a basic
neural network. For anything else, you should know how to access the
html help and read about how to do it.

## What is Torch? ##

Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to a easy and fast scripting language (Lua) and
an underlying C/C++ implementation.  You can read more about Lua
[here](http://www.lua.org).

## Installation ##

First before you can do anything, you need to install Torch7 on your
machine.  That is not described in detail here, but is instead
described in the [installation help](..:install:index).


## Checking your installation works and requiring packages ##

If you have got this far, hopefully your Torch installation works. A simple
way to make sure it does is to start Lua from the shell command line, 
and then try to start Torch:
```lua
$ torch
Try the IDE: torch -ide
Type help() for more info
Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU
Lua 5.1  Copyright (C) 1994-2008 Lua.org, PUC-Rio
t7> 
t7> x = torch.Tensor()
t7> print(x)
[torch.DoubleTensor with no dimension]

```

You might have to specify the exact path of the `torch` executable
if you installed Torch in a non-standard path.

In this example, we checked Torch was working by creating an empty
[Tensor](..:torch:tensor) and printing it on the screen.  The Tensor
is the main tool in Torch, and is used to represent vector, matrices
or higher-dimensional objects (tensors).

`torch` only preloads the basic parts of torch (including
Tensors). To see the list of all packages distributed with Torch7,
click [here](..:index).

## Getting Help ##

There are two main ways of getting help in Torch7. One way is ofcourse
the html formatted help. However, another and easier method is to use
inline help in torch interpreter. The `torch` executable also
integrates this capability. Help about any function can be accessed by
calling the `help()` function.

```lua

t7> help(torch.rand)

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
torch.rand( [res,] m [, n, k, ...])        
 y=torch.rand(n) returns a one-dimensional tensor of size n filled with 
random numbers from a uniform distribution on the interval (0,1).
 y=torch.rand(m,n) returns a mxn tensor of random numbers from a uniform 
distribution on the interval (0,1).
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

```

Even a more intuitive method is to use tab completion. Whenever any
input is entered at the `torch` prompt, one can eneter two
consecutive `TAB` characters (`double TAB`) to get the syntax
completion. Moreover entering `double TAB` at an open paranthesis
also causes the help for that particular function to be printed.

```lua

t7> torch.randn( -- enter double TAB after (
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
torch.randn( [res,] m [, n, k, ...])       
 y=torch.randn(n) returns a one-dimensional tensor of size n filled with 
random numbers from a normal distribution with mean zero and variance 
one.
 y=torch.randn(m,n) returns a mxn tensor of random numbers from a normal 
distribution with mean zero and variance one.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/  \  
t7> torch.randn(

```

## Lua Basics ##

Torch is entirely built around [Lua](http://www.lua.org/), so the first thing you have to 
know is get some basic knowledge about the language. The [online book](http://www.lua.org/docs.html)
is great for that. 

Here I'll just summarize a couple of very basic things to get you started.

### Variables ###

Creating variables is straightforward, Lua is dynamically typed language. Printing variables from the prompt is a bit misleading, you have to add the = sign before it:

```lua
t7> a = 10
t7> print(a)
10
t7> = a
10
t7> b = a + 1
t7> = b
11
```

### Lua's universal data structure: the table ###

The best thing about Lua is its consistency, and compactness. The whole language relies on a single data structure, the table, which will allow you to construct the most complex programs, with style!

  * The Lua table can be used as:
  * an array (with linearly stored values, of arbitrary types)
  * a hash-table (with hashed key/value pairs)
  * an object (which is just a hash-table)
  * a namespace (any package in Lua is a simple table, in fact the global namespace, _G, is also a table)

You already know enough about tables, let's hack around:

```lua
t7> t = {}
t7> =t
{}
t7> t = {1,2,3,4}
t7> =t
{[1] = 1
 [2] = 2
 [3] = 3
 [4] = 4}
t7> = {1,2,3,'mixed types',true}
{[1] = 1
 [2] = 2
 [3] = 3
 [4] = string : "mixed types"
 [5] = true}
t7> t =  {4,3,2,1}
```

In the example above, we've shown how to use a table as a linear array. Lua is one-based, like Matlab, so if we try to get the length of this last array created, it'll be equal to the number of elements we've put in:

```lua
t7> =#t
4
```

Ok, let's see about hash-tables now:

```lua
t7> h = {firstname='Paul', lastname='Eluard', age='117'}
t7> =h
{[firstname] = string : "Paul"
 [lastname]  = string : "Eluard"
 [age]       = string : "117"}
```

So now mixing arrays and hash-tables is easy:

```lua
t7> h = {firstname='Paul', lastname='Eluard', age='117', 1, 2, 3}
t7> =h
{[1]         = 1
 [2]         = 2
 [3]         = 3
 [firstname] = string : "Paul"
 [lastname]  = string : "Eluard"
 [age]       = string : "117"}
t7> 
```

Easy right?

So we've seen a couple of basic types already: strings, numbers, tables, booleans (true/false). There's one last type in Lua: the function. 
are first-order citizens in Lua, which means that they can be treated as regular variables. This is great, because it's the reason why we can construct very powerful data structures (such as objects) with tables:

```lua
t7> h = {firstname='Paul', lastname='Eluard', age='117',
. >      print=function(self)
. >               print(self.firstname .. ' ' .. self.lastname 
. >                     .. ' (age: ' .. self.age .. ')')
. >      end
. >     }

t7> =h
{[firstname] = string : "Paul"
 [print]     = function: 0x7f885d00c430
 [lastname]  = string : "Eluard"
 [age]       = string : "117"}
```

In this example above, we're basically storing a function at the key (hash) print. It's fairly straightforward, note that the function takes one argument, named self, which is assumed to be the object itself. The function simply concatenates the fields of the table self, and prints the whole string.

One important note: accessing fields of a table is either done using square brackets [], or the . operator. The square brackets are more general: they allow the use of arbitrary strings. In the following, we now try to access the elements of h, that we just created:

```lua
t7> h. + TAB
h.age        h.firstname  h.lastname   h.print(     

t7> = h.print
function: 0x7f885d00ec80

t7> h.print(h)
Paul Eluard (age: 117)

t7> h:print()
Paul Eluard (age: 117)
```

On the first line we type h. and then use TAB to complete and automatically explore the symbols present in h. We then print h.print, and confirm that it is indeed a function.

At the next line, we call the function h.print, and pass h as the argument (which becomes self in the body of the function). This is fairly natural, but a bit heavy to manipulate objects. Lua provides a simple shortcut, :, the column, which passes the parent table as the first argument: h:print() is strictly equivalent to h.print(h).

### Functions ###

A few more things about functions: functions in Lua are proper closures, so in combination with tables, you can use them to build complex and very flexible programs. An example of closure is given here:

```lua
myfuncs = {}
for i = 1,4 do
    local calls = 0
    myfuncs[i] = function()
        calls = calls + 1
        print('this function has been called ' .. calls .. ' times')
    end
end

t7> myfuncs[1]()
this function has been called 1 times
t7> myfuncs[1]()
this function has been called 2 times
t7> myfuncs[4]()
this function has been called 1 times
t7> myfuncs[4]()
this function has been called 2 times
t7> myfuncs[1]()
this function has been called 3 times
```

You can use such closures to create objects on the fly, that is, tables which combine functions and data to act upon. Thanks to closure, data can live in arbitrary locations (not necessarily the object's table), and simply be bound at runtime to the function's scope.

## Torch Basics: Playing with Tensors ##

Ok, now we are ready to actually do something in Torch.  Lets start by
constructing a vector, say a vector with 5 elements, and filling the
i-th element with value i. Here's how:

```lua
t7> x=torch.Tensor(5)
t7> for i=1,5 do x[i]=i; end
t7> print(x)

 1
 2
 3
 4
 5
[torch.DoubleTensor of dimension 5] 

t7>
```

However, making use of Lua's powerfull closures and functions being
first class citizens of the language, the same code could be written
in much nicer way:

```lua
t7> x=torch.Tensor(5)
t7> i=0;x:apply(function() i=i+1;return i; end)
t7> =x
 1
 2
 3
 4
 5
[torch.DoubleTensor of dimension 5]

t7> x:apply(function(x) return x^2; end)
t7> =x
  1
  4
  9
 16
 25
[torch.DoubleTensor of dimension 5]

t7> 
```

To make a matrix (2-dimensional Tensor), one simply does something
like `x=torch.Tensor(5,5)` instead:

```lua
x=torch.Tensor(5,5)
for i=1,5 do 
 for j=1,5 do 
   x[i][j]=math.random();
 end
end
```

Another way to do the same thing as the code above is provided by torch:

```lua
x=torch.rand(5,5)
```

The [torch](..:torch:maths) package contains a wide variety of commands 
for manipulating Tensors that follow rather closely the equivalent
Matlab commands. For example one can construct Tensors using the commands
[ones](..:torch:maths#torch.ones), 
[zeros](..:torch:maths#torch.zeros), 
[rand](..:torch:maths#torch.rand),
[randn](..:torch:maths#torch.randn) and
[eye](..:torch:maths#torch.eye), amongst others.

Similarly, row or column-wise operations such as 
[sum](..:torch:maths#torch.sum) and 
[max](..:torch:maths#torch.max) are called in the same way:

```lua
t7> x1=torch.rand(5,5)
t7> x2=torch.sum(x1,2); 
t7> print(x2) 
 2.3450
 2.7099
 2.5044
 3.6897
 2.4089
[torch.DoubleTensor of dimension 5x1]

t7>
```

Naturally, many BLAS operations like matrix-matrix, matrix-vector products
are implemented. We suggest everyone to install ATLAS or MKL libraries since
Torch7 can optionally take advantage with these very efficient and multi-threaded 
libraries if they are found in your system. Checkout 
[Mathematical operations using tensors.](..:torch:maths) for details.

```lua

t7> a=torch.ones(5,5)
t7> b=torch.ones(5,2)
t7> =a
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
[torch.DoubleTensor of dimension 5x5]

t7> =b
 1  1
 1  1
 1  1
 1  1
 1  1
[torch.DoubleTensor of dimension 5x2]

t7> =torch.mm(a,b)
 5  5
 5  5
 5  5
 5  5
 5  5
[torch.DoubleTensor of dimension 5x2]

```

## Types in Torch7 ##

In Torch7, different types of tensors can be used. By default, all
tensors are created using `double` type. `torch.Tensor` is a
convenience call to `torch.DoubleTensor`. One can easily switch the
default tensor type to other types, like `float`.

```lua
t7> =torch.Tensor()
[torch.DoubleTensor with no dimension]
t7> torch.setdefaulttensortype('torch.FloatTensor')
t7> =torch.Tensor()
[torch.FloatTensor with no dimension]
```

## Saving code to files, running files ##

Before we go any further, let's just review one basic thing: saving code to files, and executing them.

As Torch relies on Lua, it's best to give all your files a .lua extension. Let's generate a lua file that contains some Lua code, and then execute it:

```lua
$ echo "print('Hello World\!')" > helloworld.lua
...

$ torch helloworld.lua
...
Hello World!

$ torch
...
t7> dofile 'helloworld.lua'
Hello World!
```

That's it, you can either run programs form your shell, or from the Torch prompt. You can also run programs from the shell, and get an interactive prompt whenever an error occurs, or the program terminates (good for debugging):

```lua
$ torch -i helloworld.lua
...
Hello World!
t7>
```

We're good with all the basic things: you now know how to run code, from files or from the prompt, and write basic Lua (which is almost all Lua is!).

## Example: training a neural network ##

We will show now how to train a neural network using the [nn](..:nn:index) package
available in Torch.

### Torch basics: building a dataset using Lua tables ###

In general the user has the freedom to create any kind of structure he
wants for dealing with data.

For example, training a neural network in Torch is achieved easily by
performing a loop over the data, and forwarding/backwarding tensors
through the network. Then, the way the dataset is built is left to the
user's creativity.


However, if you want to use some convenience classes, like
[StochasticGradient](..:nn:index#nn.StochasticGradient), which basically
does the training loop for you, one has to follow the dataset
convention of these classes.  (We will discuss manual training of a
network, where one does not use these convenience classes, in a later
section.)

StochasticGradient expects as a `dataset` an object which implements
the operator `dataset[index]` and implements the method
`dataset:size()`. The `size()` methods returns the number of
examples and `dataset[i]` has to return the i-th example.

An `example` has to be an object which implements the operator
`example[field]`, where `field` often takes the value `1` (for
input features) or `2` (for corresponding labels), i.e an example is
a pair of input and output objects.  The input is usually a Tensor
(exception: if you use special kind of gradient modules, like
[table layers](..:nn:index#nn.TableLayers)). The label type depends
on the criterion. For example, the
[MSECriterion](..:nn:index#nn.MSECriterion) expects a Tensor, but the
[ClassNLLCriterion](..:nn:index#nn.ClassNLLCriterion) expects an
integer (the class).

Such a dataset is easily constructed by using Lua tables, but it could any object
as long as the required operators/methods are implemented.

Here is an example of making a dataset for an XOR type problem:
```lua
dataset={};
function dataset:size() return 100 end -- 100 examples
for i=1,dataset:size() do 
	local input= torch.randn(2);     --normally distributed example in 2d
	local output= torch.Tensor(1);
	if input[1]*input[2]>0 then    --calculate label for XOR function
		output[1]=-1;
	else
		output[1]=1;
	end
	dataset[i] = {input, output};
end
```

### Torch basics: building a neural network ###

To train a neural network we first need some data.  We can use the XOR data
we just generated in the section before.  Now all that remains is to define
our network architecture, and train it.

To use Neural Networks in Torch you have to require the 
[nn](..:nn:index) package. 
A classical feed-forward network is created with the `Sequential` object:
```lua
require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron
```

To build the layers of the network, you simply add the Torch objects 
corresponding to those layers to the _mlp_ variable created above.

The two basic objects you might be interested in first are the 
[Linear](..:nn:index#nn.Linear) and 
[Tanh](..:nn:index#nn.Tanh) layers.
The Linear layer is created with two parameters: the number of input
dimensions, and the number of output dimensions. 
So making a classical feed-forward neural network with one hidden layer with 
_HUs_ hidden units is as follows:
```lua
require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron
inputs=2; outputs=1; HUs=20;
mlp:add(nn.Linear(inputs,HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs,outputs))
```


### Torch basics: training a neural network ###

Now we're ready to train.
This is done with the following code:
```lua
criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
```

You should see printed on the screen something like this:
```lua
# StochasticGradient: training
# current error = 0.94550937745458
# current error = 0.83996744568527
# current error = 0.70880093908742
# current error = 0.58663679932706
# current error = 0.49190661630473
[..snip..]
# current error = 0.34533844015756
# current error = 0.344305927029
# current error = 0.34321901952818
# current error = 0.34206793525954
# StochasticGradient: you have reached the maximum number of iterations
```

Some other options of the _trainer_ you might be interested in are for example:
```lua
trainer.maxIteration = 10
trainer.shuffleIndices = false
```
See the nn package description of the
[StochasticGradient](..:nn:index#nn.StochasticGradient) object
for more details.


### Torch basics: testing your neural network ###

To test your network on a single example you can do this:
```lua
x=torch.Tensor(2);   -- create a test example Tensor
x[1]=0.5; x[2]=-0.5; -- set its values
pred=mlp:forward(x)  -- get the prediction of the mlp 
print(pred)          -- print it 
```

You should see that your network has learned XOR:
```lua
t7> x=torch.Tensor(2); x[1]=0.5; x[2]=0.5; print(mlp:forward(x))
-0.5886
[torch.DoubleTensor of dimension 1]

t7> x=torch.Tensor(2); x[1]=-0.5; x[2]=0.5; print(mlp:forward(x))
 0.9261
[torch.DoubleTensor of dimension 1]

t7> x=torch.Tensor(2); x[1]=0.5; x[2]=-0.5; print(mlp:forward(x))
 0.7913
[torch.DoubleTensor of dimension 1]

t7> x=torch.Tensor(2); x[1]=-0.5; x[2]=-0.5; print(mlp:forward(x))
-0.5576
[torch.DoubleTensor of dimension 1]
```

### Manual Training of a Neural Network ###

Instead of using the [StochasticGradient](..:nn:index#nn.StochasticGradient) class
you can directly make the forward and backward calls on the network yourself.
This gives you greater flexibility.
In the following code example we create the same XOR data on the fly
and train each example online.

```lua
criterion = nn.MSECriterion()  
mlp=nn.Sequential();  -- make a multi-layer perceptron
inputs=2; outputs=1; HUs=20;
mlp:add(nn.Linear(inputs,HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs,outputs))

for i = 1,2500 do
  -- random sample
  local input= torch.randn(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then  -- calculate label for XOR function
    output[1] = -1
  else
    output[1] = 1
  end

  -- feed it to the neural network and the criterion
  prediction = mlp:forward(input)
  criterion:forward(prediction, output)

  -- train over this example in 3 steps

  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()

  -- (2) accumulate gradients
  criterion_gradient = criterion:backward(prediction, output)
  mlp:backward(input, criterion_gradient)

  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
end
```

Super!

## Concluding remarks / going further ##

That's the end of this tutorial, but not the end of what you have left
to discover of Torch! To explore more of Torch, you should take a look
at the [Torch package help](..:index) which has been linked to
throughout this tutorial every time we have mentioned one of the basic
Torch object types.  The Torch library reference manual is available
[here](..:index) and the external torch packages installed on your
system can be viewed [here](..:torch:index).

We've also compiled a couple of demonstrations and tutorial scripts
that demonstrate how to train more complex models, and build gui-based
demos, and so on... All of these can be found in 
[this repo](http://github.com/andresy/torch-demos).

Good luck and have fun!

