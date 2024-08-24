//go:test trace
package kyro_test

import (
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/kyroy/kdtree"
	"github.com/kyroy/kdtree/kdrange"
	"github.com/rishitc/go-kd-tree/internal/benchmarks/dataset"
)

var (
	tracePath = filepath.Join("..", "dataset", "input_2d_trace.csv")
	trace     []kdtree.Point
)

func init() {
	if _, err := os.Stat(tracePath); errors.Is(err, os.ErrNotExist) {
		dataset.Trace2DGenerator(tracePath)
	}
	trace, _ = ReadPoint2DTrace(tracePath)
}

func BenchmarkKyroKDTreeCreation(b *testing.B) {
	var tree *kdtree.KDTree
	for i := 0; i < b.N; i++ {
		tree = kdtree.New(trace)
	}
	runtime.KeepAlive(tree)
}

func BenchmarkKyroKDTreeInsert(b *testing.B) {
	var tree *kdtree.KDTree
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		t := kdtree.New(trace[:len(trace)-1])
		e := trace[len(trace)-1]
		b.StartTimer()

		t.Insert(e)
	}
	runtime.KeepAlive(tree)
}

func BenchmarkKyroKDTreeRemove(b *testing.B) {
	var point kdtree.Point
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.New(trace)
		// Select random element to remove
		ti := rand.IntN(len(trace))
		e := trace[ti]
		b.StartTimer()

		point = tree.Remove(e)
	}
	runtime.KeepAlive(point)
}

func BenchmarkKyroKDTreeKNN(b *testing.B) {
	var points []kdtree.Point
	tree := kdtree.New(trace)
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

func BenchmarkKyroKDTreePoints(b *testing.B) {
	var points []kdtree.Point
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.New(trace)
		b.StartTimer()

		points = tree.Points()
	}
	runtime.KeepAlive(points)
}

func BenchmarkKyroKDTreeRangeSearch(b *testing.B) {
	var points []kdtree.Point
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree := kdtree.New(trace)
		b.StartTimer()

		points = tree.RangeSearch(kdrange.New(-1, 800, 0, 250))
	}
	runtime.KeepAlive(points)
}

func BenchmarkKyroKDTreeBalance(b *testing.B) {
	var tree *kdtree.KDTree
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tree = kdtree.New([]kdtree.Point{})
		for _, e := range trace {
			tree.Insert(e)
		}
		b.StartTimer()

		tree.Balance()
	}
	runtime.KeepAlive(tree)
}
