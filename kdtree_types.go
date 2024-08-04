package kdtree

import "fmt"

type RangeFunc[T Comparable[T]] func(T, int) RelativePosition

type RelativePosition int

const (
	BeforeRange RelativePosition = iota
	InRange
	AfterRange
)

var ErrTreeNotSetup = fmt.Errorf("tree is not setup, make sure you create the tree using NewTree")

type KDTree[T Comparable[T]] struct {
	dimensions int
	root       *kdNode[T]
	isSetup    bool
	zeroVal    T
	size       int
}

type kdNode[T Comparable[T]] struct {
	value T
	left  *kdNode[T]
	right *kdNode[T]
}
