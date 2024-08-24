package kyro_test

import (
	kyroyKDTree "github.com/kyroy/kdtree"
	"github.com/rishitc/go-kd-tree/internal/benchmarks"
)

func ReadPoint2DTrace(filename string) ([]kyroyKDTree.Point, error) {
	trace, err := benchmarks.Read2DTrace(filename)
	tensorTrace := make([]kyroyKDTree.Point, 0, len(trace))

	for _, e := range trace {
		tensorTrace = append(tensorTrace, &Point2D{float64(e[0]), float64(e[1])})
	}

	return tensorTrace, err
}
