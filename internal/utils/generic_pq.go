// This file is a rewrite of Go's "container/heap" implementation with the usages of interfaces replaced with generics.
// Link to Go's "container/heap" implementation: https://cs.opensource.google/go/go/+/refs/tags/go1.23.0:src/container/heap/heap.go

package internal

import "sort"

type Interface[T any] interface {
	sort.Interface
	Push(x T) // add x as element Len()
	Pop() T   // remove and return element Len() - 1.
}

func Init[V any, T Interface[V]](h T) {
	// heapify the input
	n := h.Len()
	for i := n/2 - 1; i >= 0; i-- {
		down(h, i, n)
	}
}

func Push[V any, T Interface[V]](h T, x V) {
	h.Push(x)
	up(h, h.Len()-1)
}

func Pop[V any, T Interface[V]](h T) V {
	n := h.Len() - 1
	h.Swap(0, n)
	down(h, 0, n)
	return h.Pop()
}

func Remove[V any, T Interface[V]](h T, i int) any {
	n := h.Len() - 1
	if n != i {
		h.Swap(i, n)
		if !down(h, i, n) {
			up(h, i)
		}
	}
	return h.Pop()
}

func Fix[V any, T Interface[V]](h T, i int) {
	if !down(h, i, h.Len()) {
		up(h, i)
	}
}

func up[V any, T Interface[V]](h T, j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !h.Less(j, i) {
			break
		}
		h.Swap(i, j)
		j = i
	}
}

func down[V any, T Interface[V]](h T, i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && h.Less(j2, j1) {
			j = j2 // = 2*i + 2  // right child
		}
		if !h.Less(j, i) {
			break
		}
		h.Swap(i, j)
		i = j
	}
	return i > i0
}
