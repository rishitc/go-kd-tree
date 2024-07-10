# K-D Tree

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![HitCount](https://hits.dwyl.com/rishitc/go-kd-tree.svg)](https://hits.dwyl.com/rishitc/go-kd-tree)

[![Tests](https://github.com/rishitc/go-kd-tree/actions/workflows/run-go-tests.yml/badge.svg?branch=main)](https://github.com/rishitc/go-kd-tree/actions/workflows/run-go-tests.yml)

An implementation of KD-Trees written in Go, by [Rishit Chaudhary](https://github.com/rishitc).

## Supported Operations

1. Efficiently find the nearest neighbor for a given node
1. Find the node with the minimum value in a particular dimension
1. Add a node to the KD-Tree
1. Delete a node from the KD-Tree
1. Stringify the KD-Tree to visualize it
1. Encode the tree into bytes
1. Decode the tree from bytes

**Note**:
I have used [FlatBuffers](https://flatbuffers.dev/) to encode and decode the KD-Tree.

## Tests

The tests cover all API usages and are a great place to start to understand how to use them.

## References

I've listed below all the references I've used to learn about KD-Trees working on this project.

1. [KD-Tree Nearest Neighbor Data Structure by Stable Sort](https://youtu.be/Glp7THUpGow?si=gw1s-XTOxpWHvnZ3)
    1. [Linked Java Implementation](https://bitbucket.org/StableSort/play/src/master/src/com/stablesort/kdtree/KDTree.java)
1. [Good KD Tree Visualization Fall 2019](https://courses.engr.illinois.edu/cs225/fa2019/notes/kd-tree/)
    1. [Alternate Link](https://courses.engr.illinois.edu/cs225/fa2019/resources/kd-tree/)
1. [Good KD Tree Visualization Spring 2019](https://courses.engr.illinois.edu/cs225/sp2019/notes/kd-tree/)
1. [K-D Tree: build and search for the nearest neighbor by Algokodabra](https://youtu.be/ivdmGcZo6U8?si=pW7L58qt63NYpnkO)
1. [K-d Trees - Computerphile](https://youtu.be/BK5x7IUTIyU?si=Ub1g9a601gTtLI73)
1. [Advanced Data Structures: K-D Trees by Niema Moshiri](https://youtu.be/XG4zpiJAkD4?si=UK5haTuLBahgQ2-G)
1. [Advanced Data Structures: KDT Grid Representation by Niema Moshiri](https://youtu.be/CYS2BIa79Os?si=pEYqRAGAZ7Daj3sU)
1. [Advanced Data Structures: KDT Insertion Order and Balance by Niema Moshiri](https://youtu.be/X46vqqutpBA?si=K5WMmBE7qq7aHMD-)
1. [Node Deletion in KD-Tree](https://youtu.be/DkBNF98MV1Q?si=TvFJ-sBjXBCMiSV5)
    1. The last example in this video lecture has an error. I've described and corrected this in my test case.
