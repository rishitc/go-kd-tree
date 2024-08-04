package tests

import (
	"container/heap"
	"testing"

	boundedpq "github.com/rishitc/go-kd-tree"
	types "github.com/rishitc/go-kd-tree/internal/types"
)

func TestBoundedPriorityQueue(t *testing.T) {
	table := []struct {
		name            string
		inputElems      []boundedpq.Item[types.Tensor2D] // Some items and their priorities.
		inputMaxSize    int
		expLen          int
		expOrderedElems []types.Tensor2D
	}{
		{
			name: "More number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []boundedpq.Item[types.Tensor2D]{
				{Data: types.Tensor2D{1, 2}, Priority: 3},
				{Data: types.Tensor2D{3, 5}, Priority: 2},
				{Data: types.Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 2,
			expLen:       2,
			expOrderedElems: []types.Tensor2D{
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Less number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []boundedpq.Item[types.Tensor2D]{
				{Data: types.Tensor2D{1, 2}, Priority: 3},
				{Data: types.Tensor2D{3, 5}, Priority: 2},
				{Data: types.Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 5,
			expLen:       3,
			expOrderedElems: []types.Tensor2D{
				{6, 7},
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Equal number of input elements as compared to the max allowed size of the bounded priority queue",
			inputElems: []boundedpq.Item[types.Tensor2D]{
				{Data: types.Tensor2D{1, 2}, Priority: 3},
				{Data: types.Tensor2D{3, 5}, Priority: 2},
				{Data: types.Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 3,
			expLen:       3,
			expOrderedElems: []types.Tensor2D{
				{6, 7},
				{1, 2},
				{3, 5},
			},
		},
		{
			name: "Inserting two elements with the same priority and the one inserted first should be retained",
			inputElems: []boundedpq.Item[types.Tensor2D]{
				{Data: types.Tensor2D{1, 2}, Priority: 2},
				{Data: types.Tensor2D{3, 5}, Priority: 3},
				{Data: types.Tensor2D{4, 8}, Priority: 4},
				{Data: types.Tensor2D{6, 7}, Priority: 4},
			},
			inputMaxSize: 3,
			expLen:       3,
			expOrderedElems: []types.Tensor2D{
				{4, 8},
				{3, 5},
				{1, 2},
			},
		},
	}

	for _, st := range table {
		t.Run(st.name, func(t *testing.T) {
			pq := boundedpq.NewBoundedPriorityQueue[types.Tensor2D](st.inputMaxSize)
			for _, elem := range st.inputElems {
				data, priority := elem.Data, elem.Priority
				heap.Push(&pq, boundedpq.Item[types.Tensor2D]{
					Data:     data,
					Priority: priority,
				})
			}

			if pq.Len() != st.expLen {
				t.Errorf("Expected size to be %v, got %v", st.expLen, pq.Len())
			}

			for i := range st.expLen {
				headElem := pq.Peek().(boundedpq.Item[types.Tensor2D]).Data
				expElem := st.expOrderedElems[i]
				if expElem[0] != headElem[0] && expElem[1] != headElem[1] {
					t.Errorf("Expected next element to be %v, got %v", expElem, headElem)
				}
				heap.Pop(&pq)
			}
		})
	}
}

