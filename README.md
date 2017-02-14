#READ ME#


This is a simple example implementation of the Inverse Compositional[1] method 
which is a variant on the Lucas-Kanade[2] algorithm for image registration. 
The difference is that it keeps the Gauss-Newton approximation to the Hessian fixed across iterations.

For extra credit I've also thrown in a Huber[3] loss function to make it robust to outliers but have not really played much with it. 
Set it to 1 for regular quadratic loss

But why is it called sim2-alignment?  
Because it estimates the parameters for translation, 
rotation and scale, i.e. a similarity transform in 2D

Dependencies:

This program depends on Eigen3, OpenCV and cmake (and optionally OpenMP).
Compilation instructions:

```sh
$ mkdir build && cd build
$ cmake ..
$ make
```
Running the program:

from the main directory:

```sh
./bin/align lena_src.png lena.png 150 50
./bin/align cameraman_src.png cameraman.png
./bin/align cat_src.png cat.png 160 160
```

The first and third images are more challenging, but given the initialization (offset in rows and columns respectively) they converge.

##References:
 - [1] Baker, Simon, and Iain Matthews. "Lucas-kanade 20 years on: A unifying framework." International journal of computer vision 56.3 (2004): 221-255.

 - [2] Lucas, Bruce D., and Takeo Kanade. "An iterative image registration technique with an application to stereo vision." IJCAI. Vol. 81. No. 1. 1981.

 - [3] Huber, Peter J. "Robust estimation of a location parameter." The Annals of Mathematical Statistics 35.1 (1964): 73-101.
 
 ##(c) Daniel Ricão Canelhas
