# Vertex-CNN

We introduce a vertex-based graph convolutional neural network (vertex-CNN) for analyzing structured data on graphs. 
We represent graphs using semi-regular triangulated meshes in which each vertex has 6 connected neighbors. We generalize 
classical CNN defined on equi-spaced grids to that defined on semi-regular triangulated meshes by introducing main building 
blocks of the CNN, including convolution, down-sampling, and pooling, on a vertex domain.  By exploiting the regularity of 
semi-regular meshes in terms of vertex connections, the proposed vertex-CNN keeps the inherent properties of classical CNN 
in a Euclidean space, such as shift-invariance and down-sampling at a rate of 2, 4, etc.

Reference:

Chaoqiang Liu, Hui Ji, Anqi Qiu, "Fast Vertex-Based Graph Convolutional Neural Network and its Application to Brain Images".
