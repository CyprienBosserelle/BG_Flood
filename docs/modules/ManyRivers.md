# Efficient run time with many rivers
BG_Flood has been designed to handle a very large number of river injection with minimal impact on the runtime.

# The problem

In a naive implementation the model loops through all the river injection, each kernels are launched in serial for each river. whith the kernel launch, the whole GPU is then held up for updating a few cells for each rivers at a time. This is simple, it is safe when multiple river affect the same cells but extremely inefficient when you have hundreds of injections.


## Yes this is an issue
For example, in a crude test model, the runtime for 675 river is 10x than for each river. In that test model, the time step is roughly the same and the number of wet cells are the same. This test may not be the worse case scenario but a pretty good illustration of the issue.

| n rivers | Run times |
| -------- | ------- |
| 1 | 6    |
| 10 | 7    |
| 100    | 14    |
| 675 | 71 | 


# How to fix this
This is a classic GPU problem when you need to flatten a loop to make it work efficiently. This is not so trivial because BG_Flood needs to retain its ability to inject multiple rivers at the same place without creating a race issue (when 2 or more processes try to add a value at the same cell). Multiple river at the same location often occur in BG_Flood if the river discharges are obtained from a hydrological model. However, Most river injection are in blocks distant from each other so some flattening can happen.

## More details
There is still a iteration loop for the maximum number of river per block but each iteration is flattened and memory efficient wher only affected blocks are called in the kernel and as many river as unique blocks are called at once.


# Results
Using the same setup as above but rerunning each scenarios. (Waitaki domain, single level (256m res). 3600 s run ). We can see that the new implementation is very efficient for a realistic setup and the number of river injection does not significantly slow down the simulation. 


| n rivers | Run times old code |Run times new code|
| -------- | ------- |------- |
| 1 | 5  | 5 |
| 10 | 5 | 6|
| 100    | 14  | 7| 
| 675 | 73| 9-10 |
