package kdtree

import "fmt"

type Comparable[T any] interface {
	fmt.Stringer
	Order(rhs T, dim int) int
	Dist(rhs T) int
	DistDim(rhs T, dim int) int
	Encode() []byte
}
