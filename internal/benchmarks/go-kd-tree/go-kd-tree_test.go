//go:test trace
package gokdtree_test

import (
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
	"github.com/rishitc/go-kd-tree/internal/benchmarks/dataset"
	"github.com/rishitc/go-kd-tree/internal/types"
)

const dimensions2DCount = 2

var (
	tracePath = filepath.Join("..", "dataset", "input_2d_trace.csv")
	trace     []types.Tensor2D
)

func init() {
	if _, err := os.Stat(tracePath); errors.Is(err, os.ErrNotExist) {
		dataset.Trace2DGenerator(tracePath)
	}
	trace, _ = ReadTensor2DTrace(tracePath)
}

func BenchmarkGoKDTreeCreation(b *testing.B) {
	var tree *kdtree.KDTree[types.Tensor2D]
	for i := 0; i < b.N; i++ {
		tree = kdtree.NewKDTreeWithValues(dimensions2DCount, trace)
	}
	runtime.KeepAlive(tree)
}

func BenchmarkGoKDTreeInsert(b *testing.B) {
	var tree *kdtree.KDTree[types.Tensor2D]
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		t := kdtree.NewKDTreeWithValues(dimensions2DCount, trace[:len(trace)-1])
		e := trace[len(trace)-1]
		b.StartTimer()

		t.Insert(e)
	}
	runtime.KeepAlive(tree)
}

func BenchmarkGoKDTreeRemove(b *testing.B) {
	var point types.Tensor2D
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.NewKDTreeWithValues(dimensions2DCount, trace)
		// Select random element to remove
		ti := rand.IntN(len(trace))
		e := trace[ti]
		b.StartTimer()

		tree.Remove(e)
	}
	runtime.KeepAlive(point)
}

func BenchmarkGoKDTreeKNN(b *testing.B) {
	var points []types.Tensor2D
	tree := kdtree.NewKDTreeWithValues(dimensions2DCount, trace)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ti := rand.IntN(len(trace))
		e := trace[ti]
		b.StartTimer()

		points = tree.KNN(e, 100)
	}
	runtime.KeepAlive(points)
}

func BenchmarkGoKDTreeValues(b *testing.B) {
	var points []types.Tensor2D
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.NewKDTreeWithValues(dimensions2DCount, trace)
		b.StartTimer()

		points = tree.Values()
	}
	runtime.KeepAlive(points)
}

func BenchmarkGoKDTreeRangeSearch(b *testing.B) {
	var points []types.Tensor2D
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.NewKDTreeWithValues(dimensions2DCount, trace)
		b.StartTimer()

		points = tree.RangeSearch(func(td types.Tensor2D, i int) kdtree.RelativePosition {
			const (
				xs = -1
				xe = 801
				ys = 0
				ye = 251
			)
			switch i {
			case -1:
				if x, y := td[0], td[1]; xs <= x && x < xe && ys <= y && y < ye {
					return kdtree.InRange
				}
				return kdtree.AfterRange
			case 0:
				if x := td[0]; x < xs {
					return kdtree.BeforeRange
				} else if x >= xe {
					return kdtree.AfterRange
				} else {
					return kdtree.InRange
				}
			case 1:
				if y := td[1]; y < ys {
					return kdtree.BeforeRange
				} else if y >= ye {
					return kdtree.AfterRange
				} else {
					return kdtree.InRange
				}
			}
			return kdtree.AfterRange
		})
	}
	runtime.KeepAlive(points)
}

func BenchmarkGoKDTreeBalance(b *testing.B) {
	var tree *kdtree.KDTree[types.Tensor2D]
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree = kdtree.NewKDTreeWithValues(dimensions2DCount, []types.Tensor2D{})
		for _, e := range trace {
			tree.Insert(e)
		}
		b.StartTimer()

		tree.Balance()
	}
	runtime.KeepAlive(tree)
}
