package kdtree

import (
	"slices"

	heap "github.com/rishitc/go-kd-tree/internal/utils"
)

type Item[T Comparable[T]] struct {
	Data     *T
	Priority int
}

type BoundedPriorityQueue[T Comparable[T]] struct {
	data     []Item[T]
	capacity int
}

func NewBoundedPriorityQueue[T Comparable[T]](maxSize int) BoundedPriorityQueue[T] {
	return BoundedPriorityQueue[T]{
		data:     make([]Item[T], 0, maxSize),
		capacity: maxSize,
	}
}

func (pq BoundedPriorityQueue[T]) Len() int { return len(pq.data) }

func (pq BoundedPriorityQueue[T]) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq.data[i].Priority > pq.data[j].Priority
}

func (pq BoundedPriorityQueue[T]) Swap(i, j int) {
	pq.data[i], pq.data[j] = pq.data[j], pq.data[i]
}

func (pq *BoundedPriorityQueue[T]) Push(item Item[T]) {
	isFull := pq.Len() == pq.capacity

	if isFull {
		if pq.data[0].Priority <= item.Priority {
			return
		}
		heap.Pop(pq)
	}
	pq.data = append(pq.data, item)
}

func (pq *BoundedPriorityQueue[T]) Pop() Item[T] {
	n := pq.Len()
	item := pq.data[n-1]
	pq.data = slices.Delete(pq.data, n-1, n)
	return item
}

func (pq *BoundedPriorityQueue[T]) Peek() Item[T] {
	return pq.data[0]
}

func (pq *BoundedPriorityQueue[T]) Capacity() int {
	return pq.capacity
}
