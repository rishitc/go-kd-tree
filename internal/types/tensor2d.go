package types

import (
	"encoding/json"
	"fmt"

	"golang.org/x/exp/rand"
)

type Tensor2D [2]int

func (lhs Tensor2D) Order(rhs Tensor2D, dim int) int {
	const dimensions = 2
	l := ([2]int)(lhs)
	r := ([2]int)(rhs)
	for i := 0; i < dimensions; i++ {
		if l[dim] < r[dim] {
			return -1
		} else if l[dim] > r[dim] {
			return 1
		}
		dim = (dim + 1) % dimensions
	}
	return 0
}

func (lhs Tensor2D) DistDim(rhs Tensor2D, dim int) int {
	l := ([2]int)(lhs)
	r := ([2]int)(rhs)
	return (l[dim] - r[dim]) * (l[dim] - r[dim])
}

func (lhs Tensor2D) Dist(rhs Tensor2D) int {
	l := ([2]int)(lhs)
	r := ([2]int)(rhs)
	sumOfSquaredDistances := 0

	for i := 0; i < len(l); i++ {
		distance := l[i] - r[i]
		squaredDist := distance * distance
		sumOfSquaredDistances += squaredDist
	}

	return sumOfSquaredDistances
}

func (lhs Tensor2D) String() string {
	return fmt.Sprintf("[%d, %d]", lhs[0], lhs[1])
}

func (lhs Tensor2D) Encode() []byte {
	b, _ := json.Marshal(lhs)
	return b
}

func DecodeTensor2D(bytes []byte) Tensor2D {
	var v Tensor2D
	if err := json.Unmarshal(bytes, &v); err != nil {
		msg := fmt.Sprintf("JSON unmarshal failed %v", err)
		panic(msg)
	}
	return v
}

func GenerateRandomTensor2DSlice(n int) []Tensor2D {
	arrays := make([]Tensor2D, n)
	r := rand.New(rand.NewSource(43))
	for i := range arrays {
		array := Tensor2D{r.Int(), r.Int()}
		arrays[i] = array
	}
	return arrays
}
