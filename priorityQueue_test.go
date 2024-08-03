package kdtree_test

import (
	"container/heap"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
)

// This example creates a PriorityQueue with some items, adds and manipulates an item,
// and then removes the items in priority order.
func TestBoundedPriorityQueue(t *testing.T) {
	table := []struct {
		name            string
		inputElems      []kdtree.Item[Tensor2D] // Some items and their priorities.
		inputMaxSize    int
		expLen          int
		expOrderedElems []Tensor2D
	}{
		{
			name: "More number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []kdtree.Item[Tensor2D]{
				{Data: Tensor2D{1, 2}, Priority: 3},
				{Data: Tensor2D{3, 5}, Priority: 2},
				{Data: Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 2,
			expLen:       2,
			expOrderedElems: []Tensor2D{
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Less number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []kdtree.Item[Tensor2D]{
				{Data: Tensor2D{1, 2}, Priority: 3},
				{Data: Tensor2D{3, 5}, Priority: 2},
				{Data: Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 5,
			expLen:       3,
			expOrderedElems: []Tensor2D{
				{6, 7},
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Equal number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []kdtree.Item[Tensor2D]{
				{Data: Tensor2D{1, 2}, Priority: 3},
				{Data: Tensor2D{3, 5}, Priority: 2},
				{Data: Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 3,
			expLen:       3,
			expOrderedElems: []Tensor2D{
				{6, 7},
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Inserting two elements with the same priority and the one inserted first should be retained",
			inputElems: []kdtree.Item[Tensor2D]{
				{Data: Tensor2D{1, 2}, Priority: 2},
				{Data: Tensor2D{3, 5}, Priority: 3},
				{Data: Tensor2D{4, 8}, Priority: 4},
				{Data: Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 3,
			expLen:       3,
			expOrderedElems: []Tensor2D{
				{4, 8},
				{3, 5},
				{1, 2},
			},
		},
	}

	for _, st := range table {
		t.Run(st.name, func(t *testing.T) {
			pq := kdtree.NewBoundedPriorityQueue[Tensor2D](st.inputMaxSize)
			for _, elem := range st.inputElems {
				data, priority := elem.Data, elem.Priority
				heap.Push(&pq, kdtree.Item[Tensor2D]{
					Data:     data,
					Priority: priority,
				})
			}

			if pq.Len() != st.expLen {
				t.Errorf("Expected size to be %v, got %v", st.expLen, pq.Len())
			}

			for i := range st.expLen {
				headElem := pq.Peek().(kdtree.Item[Tensor2D]).Data
				expElem := st.expOrderedElems[i]
				if expElem[0] != headElem[0] && expElem[1] != headElem[1] {
					t.Errorf("Expected next element to be %v, got %v", expElem, headElem)
				}
				heap.Pop(&pq)
			}
		})
	}
}

