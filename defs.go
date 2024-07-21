package kdtree

import "fmt"

type Relation int

const (
	Lesser Relation = iota
	Equal
	Greater
)

type RelativePosition int

const (
	BeforeRange RelativePosition = iota
	InRange
	AfterRange
)

var ErrTreeNotSetup = fmt.Errorf("tree is not setup, make sure you create the tree using NewTree")

type Comparable[T any] interface {
	fmt.Stringer
	Order(rhs T, dim int) Relation
	Dist(rhs T) int
	DistDim(rhs T, dim int) int
	Encode() []byte
}

type KDTree[T Comparable[T]] struct {
	dimensions int
	root       *kdNode[T]
	isSetup    bool
	zeroVal    T
	sz         int
}

type kdNode[T Comparable[T]] struct {
	value T
	left  *kdNode[T]
	right *kdNode[T]
}
