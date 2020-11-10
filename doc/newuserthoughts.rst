A place for my thoughts as a new user
==================

Installation
---------------------------
Easy, see <reference> to install using emirge


What it the type of this thing (what are these objects)
---------------------------
Objects are great and have a lot of usage hence why they are heavily leveraged in the code.
However they can be confusing to a new user (especially one also learning a new language--python).
Here are few examples to help elucidate the basics, and how they are used in mirgecom.

First conider taking the gradiant a solution variable, q

    r = discr.weak_grad(q)
    
Here we pass a DOFArray (an array of the nodal solution values) and we get back an object array of DOFArrays.
The gradient of a scalar is a vector (of length dim--the dimensionally of your solution space).
Hence, the returned value is a object array of size (dim,) such that each component contains a DOFArray.

Additionally consider for example

    q = discr.weak_div(r)
    
The divergence requires a vector to operate on, and hence this function expects an object array of size (dim,) of DOFArrays.
It then returns only a DOFArray, as the result is a scalar.

Now let's consider something more complicated

     u = obj_array_vectorize(discr.weak_div, v)

obj_array_vectorize takes a function--here discr.weak_div--and an object array--here v.
It then operates the specified function on each component of the object array (v[i]), and returns an object array of the results of each component.

Hence

    u[i] = discr.weak_div(v[i])

So, what is v in the above example? 
Well v[i] must be an object array of size (dim,) of DOFArrays as required by weak_div.
Then, v must be an object array of object arrays (size (dim,)) of DOFArrays.
The returned result is than only an object array of DOFArrays, as weak_div returns an DOFArray not a object array.


You can find the exact specifics of each function from their respective documentation, for example here is grudge's related to the previous examples:

https://documen.tician.de/grudge/discretization.html?highlight=weak_div#grudge.eager.EagerDGDiscretization

