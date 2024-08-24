package types

import (
	"encoding/json"
	"fmt"
)

type Tensor3D [3]int

func (lhs Tensor3D) Order(rhs Tensor3D, dim int) int {
	const dimensions = 3
	l := ([3]int)(lhs)
	r := ([3]int)(rhs)

	// Optimization from the paper: Optimized Super Key Comparison
	if l[dim] < r[dim] {
		return -1
	} else if l[dim] > r[dim] {
		return 1
	}
	dim = (dim + 1) % dimensions
	for i := 0; i < dimensions-1; i++ {
		if l[dim] < r[dim] {
			return -1
		} else if l[dim] > r[dim] {
			return 1
		}
		dim = (dim + 1) % dimensions
	}
	return 0
}

func (lhs Tensor3D) DistDim(rhs Tensor3D, dim int) int {
	l := ([3]int)(lhs)
	r := ([3]int)(rhs)
	return (l[dim] - r[dim]) * (l[dim] - r[dim])
}

func (lhs Tensor3D) Dist(rhs Tensor3D) int {
	l := ([3]int)(lhs)
	r := ([3]int)(rhs)
	sumOfSquaredDistances := 0

	for i := 0; i < len(l); i++ {
		distance := l[i] - r[i]
		squaredDist := distance * distance
		sumOfSquaredDistances += squaredDist
	}

	return sumOfSquaredDistances
}

func (lhs Tensor3D) String() string {
	return fmt.Sprintf("[%d, %d, %d]", lhs[0], lhs[1], lhs[2])
}

func (lhs Tensor3D) Encode() []byte {
	b, _ := json.Marshal(lhs)
	return b
}

func DecodeTensor3D(bytes []byte) Tensor3D {
	var v Tensor3D
	if err := json.Unmarshal(bytes, &v); err != nil {
		msg := fmt.Sprintf("JSON unmarshal failed %v", err)
		panic(msg)
	}
	return v
}
