package gokdtree_test

import (
	"github.com/rishitc/go-kd-tree/internal/benchmarks"
	"github.com/rishitc/go-kd-tree/internal/types"
)

func ReadTensor2DTrace(filename string) ([]types.Tensor2D, error) {
	trace, err := benchmarks.Read2DTrace(filename)
	tensorTrace := make([]types.Tensor2D, 0, len(trace))

	for _, e := range trace {
		tensorTrace = append(tensorTrace, types.Tensor2D(e))
	}

	return tensorTrace, err
}
