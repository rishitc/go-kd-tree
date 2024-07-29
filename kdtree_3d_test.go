package kdtree_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
)

type Tensor3D [3]int

func (lhs Tensor3D) Order(rhs Tensor3D, dim int) kdtree.Relation {
	const dimensions = 3
	l := ([3]int)(lhs)
	r := ([3]int)(rhs)

	// Optimization from the paper: Optimized Super Key Comparison
	if l[dim] < r[dim] {
		return kdtree.Lesser
	} else if l[dim] > r[dim] {
		return kdtree.Greater
	}
	dim = (dim + 1) % dimensions
	for i := 0; i < dimensions-1; i++ {
		if l[dim] < r[dim] {
			return kdtree.Lesser
		} else if l[dim] > r[dim] {
			return kdtree.Greater
		}
		dim = (dim + 1) % dimensions
	}
	return kdtree.Equal
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

func decodeTensor3D(bytes []byte) Tensor3D {
	var v Tensor3D
	if err := json.Unmarshal(bytes, &v); err != nil {
		msg := fmt.Sprintf("JSON unmarshal failed %v", err)
		panic(msg)
	}
	return v
}

func generateRandomTensor3DSlice(n int) []Tensor3D {
	arrays := make([]Tensor3D, n)
	r := rand.New(rand.NewSource(43))
	for i := range arrays {
		array := Tensor3D{r.Int(), r.Int(), r.Int()}
		arrays[i] = array
	}
	return arrays
}

func Test3DNearestNeighbor1(t *testing.T) {
	const dimensions = 3
	ps := []Tensor3D{
		{2, 3, 3},
		{5, 4, 2},
		{9, 6, 7},
		{4, 7, 9},
		{8, 1, 5},
		{7, 2, 6},
		{9, 4, 1},
		{8, 4, 2},
		{9, 7, 8},
		{6, 3, 1},
		{3, 4, 5},
		{1, 6, 8},
		{9, 5, 3},
		{2, 1, 3},
		{8, 7, 6},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor3D
	}{
		{
			input:    [3]int{7, 2, 5},
			expected: [3]int{7, 2, 6},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test3DNearestNeighbor2(t *testing.T) {
	const dimensions = 2
	ps := []Tensor3D{
		{5, 4},
		{3, 1},
		{2, 6},
		{8, 7},
		{10, 2},
		{13, 3},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor3D
	}{
		{
			input:    [3]int{9, 4},
			expected: [3]int{10, 2},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test3DNearestNeighbor3(t *testing.T) {
	const dimensions = 2
	ps := []Tensor3D{
		{207, 313},
		{70, 721},
		{343, 858},
		{615, 40},
		{751, 177},
		{479, 449},
		{888, 585},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor3D
	}{
		{
			input:    [3]int{438, 681},
			expected: [3]int{343, 858},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test3DNearestNeighbor4(t *testing.T) {
	const dimensions = 2
	ps := []Tensor3D{
		{272, 59},
		{259, 189},
		{481, 144},
		{915, 157},
		{139, 310},
		{913, 276},
		{43, 480},
		{281, 467},
		{622, 410},
		{821, 386},
		{136, 615},
		{445, 585},
		{260, 685},
		{592, 715},
		{749, 683},
		{163, 826},
		{438, 828},
		{571, 839},
		{662, 798},
		{879, 810},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor3D
	}{
		{
			input:    [3]int{298, 825},
			expected: [3]int{163, 826},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test3DTreeEncodeDecode(t *testing.T) {
	const dimensions = 3
	treeNodes := kdtree.NewKDNode(Tensor3D{7, 77, 10}).
		SetLeft(
			kdtree.NewKDNode(Tensor3D{0, 5, 1}).
				SetLeft(
					kdtree.NewKDNode(Tensor3D{1, 2, 3}),
				).
				SetRight(
					kdtree.NewKDNode(Tensor3D{0, 50, 1}).
						SetRight(
							kdtree.NewKDNode(Tensor3D{1, 7, 1}),
						),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor3D{9, 16, 4}).
				SetLeft(
					kdtree.NewKDNode(Tensor3D{90, 0, 1}).
						SetRight(
							kdtree.NewKDNode(Tensor3D{44, 5, 14}),
						),
				).
				SetRight(
					kdtree.NewKDNode(Tensor3D{10, 20, 1}).
						SetRight(
							kdtree.NewKDNode(Tensor3D{31, 42, 49}),
						),
				),
		)
	expectedTree := kdtree.NewTestKDTree(dimensions, treeNodes)
	encodedTreeBytes := expectedTree.Encode()
	tree := kdtree.NewKDTreeFromBytes(encodedTreeBytes, decodeTensor3D)
	if !kdtree.IdenticalTrees(tree, expectedTree) {
		t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", expectedTree, tree)
	}
}
