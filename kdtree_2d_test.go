package kdtree_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
	"github.com/stretchr/testify/assert"
)

type Tensor2D [2]int

func (lhs Tensor2D) Order(rhs Tensor2D, dim int) kdtree.Relation {
	const dimensions = 2
	l := ([2]int)(lhs)
	r := ([2]int)(rhs)
	for i := 0; i < dimensions; i++ {
		if l[dim] < r[dim] {
			return kdtree.Lesser
		} else if l[dim] > r[dim] {
			return kdtree.Greater
		}
		dim = (dim + 1) % dimensions
	}
	return kdtree.Equal
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

func decodeTensor2D(bytes []byte) Tensor2D {
	var v Tensor2D
	if err := json.Unmarshal(bytes, &v); err != nil {
		msg := fmt.Sprintf("JSON unmarshal failed %v", err)
		panic(msg)
	}
	return v
}

func generateRandomTensor2DSlice(n int) []Tensor2D {
	arrays := make([]Tensor2D, n)
	r := rand.New(rand.NewSource(43))
	for i := range arrays {
		array := Tensor2D{r.Int(), r.Int()}
		arrays[i] = array
	}
	return arrays
}

func Test2DNearestNeighbor1(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{3, 2},
		{5, 8},
		{6, 1},
		{9, 0},
		{4, 4},
		{1, 1},
		{2, 2},
		{8, 7},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor2D
	}{
		{
			input:    [2]int{-1, -1},
			expected: [2]int{1, 1},
		},
		{
			input:    [2]int{0, 0},
			expected: [2]int{1, 1},
		},
		{
			input:    [2]int{4, 2},
			expected: [2]int{3, 2},
		},
		{
			input:    [2]int{6, 6},
			expected: [2]int{5, 8}, // []int{8, 7} is also correct as it is the same distance away
		},
		{
			input:    [2]int{6, 1},
			expected: [2]int{6, 1},
		},
		{
			input:    [2]int{9, 0},
			expected: [2]int{9, 0},
		},
		{
			input:    [2]int{7, 2},
			expected: [2]int{6, 1},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DNearestNeighbor2(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{5, 4},
		{3, 1},
		{2, 6},
		{8, 7},
		{10, 2},
		{13, 3},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor2D
	}{
		{
			input:    [2]int{9, 4},
			expected: [2]int{10, 2},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DNearestNeighbor3(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
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
		input, expected Tensor2D
	}{
		{
			input:    [2]int{438, 681},
			expected: [2]int{343, 858},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DNearestNeighbor4(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
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
		input, expected Tensor2D
	}{
		{
			input:    [2]int{298, 825},
			expected: [2]int{163, 826},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DNearestNeighbor5(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{50, 50},
		{80, 40},
		{10, 60},
		{51, 38},
		{48, 38},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected Tensor2D
	}{
		{
			input:    [2]int{40, 40},
			expected: [2]int{48, 38},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func tensor2DSortFunc(a, b Tensor2D) int {
	if a[0] != b[0] {
		return a[0] - b[0]
	} else {
		return a[1] - b[1]
	}
}

func Test2DKNearestNeighbor1(t *testing.T) {
	const dimensions = 2
	type input struct {
		p Tensor2D
		k int
	}
	ps := []Tensor2D{
		{50, 50},
		{10, 25},
		{40, 20},
		{25, 80},
		{70, 70},
		{60, 10},
		{60, 90},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := map[string]struct {
		input    input
		expected []Tensor2D
	}{
		"Find the 2 closest neighbors to a point that is not in the KD tree.": {
			input:    input{p: [2]int{25, 25}, k: 2},
			expected: []Tensor2D{{40, 20}, {10, 25}},
		},
		"The closest neighbor to a point that is in the KD tree.": {
			input:    input{p: [2]int{60, 90}, k: 1},
			expected: []Tensor2D{{60, 90}},
		},
		"The three closest neighbors to a point that is in the KD tree.": {
			input:    input{p: [2]int{70, 70}, k: 3},
			expected: []Tensor2D{{50, 50}, {60, 90}, {70, 70}},
		},
	}
	for name, st := range testTable {
		t.Run(name, func(t *testing.T) {
			nns := tree.KNearestNeighbor(st.input.p, st.input.k)
			slices.SortFunc(nns, tensor2DSortFunc)
			slices.SortFunc(st.expected, tensor2DSortFunc)
			for i := range len(st.expected) {
				if st.expected[i][0] != nns[i][0] || st.expected[i][1] != nns[i][1] {
					t.Fatalf("Expected point: %v, got %v", st.expected[i], nns[i])
				}
			}
		})
	}
}

func Test2DNodeAddition1(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{50, 50},
		{80, 40},
		{10, 60},
		{51, 38},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	v := Tensor2D{48, 38}
	if ok := tree.Add(v); !ok {
		t.Fatalf("Expected value %s to be absent from tree, got that it was present already", v)
	}
	testTable := []struct {
		input, expected Tensor2D
	}{
		{
			input:    [2]int{40, 40},
			expected: [2]int{48, 38},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DNodeAddition2(t *testing.T) {
	const dimensions = 2
	tree := kdtree.NewKDTreeWithValues(dimensions, []Tensor2D{})
	ps := []Tensor2D{
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
	for _, v := range ps {
		tree.Add(v)
	}
	testTable := []struct {
		input, expected Tensor2D
	}{
		{
			input:    [2]int{298, 825},
			expected: [2]int{163, 826},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.NearestNeighbor(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DFindMin1(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{35, 90},
		{60, 80},
		{51, 75},
		{70, 70},
		{50, 50},
		{25, 40},
		{10, 30},
		{1, 10},
		{55, 1},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input    int
		expected Tensor2D
	}{
		{
			input:    0,
			expected: Tensor2D{1, 10},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.FindMin(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DFindMin2(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{35, 90},
		{60, 80},
		{51, 75},
		{70, 70},
		{50, 50},
		{25, 40},
		{10, 30},
		{1, 10},
		{55, 1},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input    int
		expected Tensor2D
	}{
		{
			input:    1,
			expected: Tensor2D{55, 1},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.FindMin(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DFindMax1(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{35, 90},
		{60, 80},
		{51, 75},
		{70, 70},
		{50, 50},
		{25, 40},
		{10, 30},
		{1, 10},
		{55, 1},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input    int
		expected Tensor2D
	}{
		{
			input:    0,
			expected: Tensor2D{70, 70},
		},
		{
			input:    1,
			expected: Tensor2D{35, 90},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.FindMax(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DFindMax2(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{35, 90},
		{60, 80},
		{51, 75},
		{70, 70},
		{50, 50},
		{25, 40},
		{10, 30},
		{1, 10},
		{55, 1},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input    int
		expected Tensor2D
	}{
		{
			input:    1,
			expected: Tensor2D{55, 1},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.FindMin(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DDeleteAllNodesInTree(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{5, 6})
	tree := kdtree.NewTestKDTree(2, treeNodes)

	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{5, 6},
			expected: kdtree.NewTestKDTree[Tensor2D](2, nil),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=YhQLGxiH7BbG9D8s&t=37
func Test2DDeleteLeafNode(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{40, 60}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{30, 40}),
				),
		)
	tree := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{40, 60}),
		)
	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{30, 40},
			expected: kdtree.NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=-tQZZtNASyMXnhNc&t=90
func Test2DDeleteNodeWithRightSubtree(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{20, 15}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{40, 60}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{30, 40}).
						SetLeft(
							kdtree.NewKDNode(Tensor2D{28, 17}),
						),
				),
		)
	tree := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes := kdtree.NewKDNode(Tensor2D{28, 17}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{20, 15}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{40, 60}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{30, 40}),
				),
		)
	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{25, 50},
			expected: kdtree.NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=v-TuZNV9YiTmCOFg&t=189
func Test2DDeleteNodeWithLeftSubtreeOnly(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{20, 15}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{40, 60}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{30, 40}).
						SetLeft(
							kdtree.NewKDNode(Tensor2D{28, 47}),
						),
				),
		)
	tree := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{20, 15}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{30, 40}).
				SetRight(
					kdtree.NewKDNode(Tensor2D{28, 47}),
				),
		)
	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{40, 60},
			expected: kdtree.NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

func Test2DDeleteNode1(t *testing.T) {
	const dimensions = 2
	ps := []Tensor2D{
		{5, 6},
		{4, 10},
		{4, 20},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	expTree := kdtree.NewKDNode(Tensor2D{4, 20}).SetLeft(kdtree.NewKDNode(Tensor2D{4, 10}))

	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{5, 6},
			expected: kdtree.NewTestKDTree(2, expTree),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}
func Test2DDeleteNode2(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{5, 6}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{4, 10}).
				SetRight(
					kdtree.NewKDNode(Tensor2D{4, 20}),
				),
		)
	tree := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes := kdtree.NewKDNode(Tensor2D{4, 10}).
		SetRight(
			kdtree.NewKDNode(Tensor2D{4, 20}),
		)
	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{5, 6},
			expected: kdtree.NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=Wl-uNf_PSVaoa2D1&t=342
// The example has a mistake: The example KD-Tree is not valid because the node (60, 10) must be the left
// child of node (70, 20), and not the right child of node (70, 20).
// However, the resultant tree shown in the video lecture is correct and has been used in this test case.
// The below test case, has been created using the corrected valid KD-Tree as input
func Test2DDeleteNode3(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{35, 60}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{20, 45}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{10, 35}).
						SetRight(
							kdtree.NewKDNode(Tensor2D{20, 20}),
						),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{60, 80}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{80, 40}).
						SetLeft(
							kdtree.NewKDNode(Tensor2D{50, 30}).
								SetLeft(
									kdtree.NewKDNode(Tensor2D{70, 20}).
										SetLeft(
											kdtree.NewKDNode(Tensor2D{60, 10}),
										),
								),
						).
						SetRight(
							kdtree.NewKDNode(Tensor2D{90, 60}),
						),
				),
		)
	tree := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes := kdtree.NewKDNode(Tensor2D{50, 30}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{20, 45}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{10, 35}).
						SetRight(
							kdtree.NewKDNode(Tensor2D{20, 20}),
						),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{60, 80}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{80, 40}).
						SetLeft(
							kdtree.NewKDNode(Tensor2D{60, 10}).
								SetRight(
									kdtree.NewKDNode(Tensor2D{70, 20}),
								),
						).
						SetRight(
							kdtree.NewKDNode(Tensor2D{90, 60}),
						),
				),
		)
	testTable := []struct {
		input    Tensor2D
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    Tensor2D{35, 60},
			expected: kdtree.NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !kdtree.IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

func Test2DTreeEncodeDecode(t *testing.T) {
	const dimensions = 2
	treeNodes := kdtree.NewKDNode(Tensor2D{4, 4}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{2, 2}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{1, 1}),
				).
				SetRight(
					kdtree.NewKDNode(Tensor2D{3, 2}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{6, 1}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{9, 0}),
				).
				SetRight(
					kdtree.NewKDNode(Tensor2D{5, 8}).
						SetRight(
							kdtree.NewKDNode(Tensor2D{8, 7}),
						),
				),
		)
	expectedTree := kdtree.NewTestKDTree(dimensions, treeNodes)
	encodedTreeBytes := expectedTree.Encode()
	tree := kdtree.NewKDTreeFromBytes(encodedTreeBytes, decodeTensor2D)
	if !kdtree.IdenticalTrees(tree, expectedTree) {
		t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", expectedTree, tree)
	}
}

func Test2DTreeBalance(t *testing.T) {
	treeNodes := kdtree.NewKDNode(Tensor2D{20, 15}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{3, 25}),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{30, 40}).
				SetRight(
					kdtree.NewKDNode(Tensor2D{25, 50}).
						SetRight(
							kdtree.NewKDNode(Tensor2D{28, 47}).
								SetRight(
									kdtree.NewKDNode(Tensor2D{40, 60}),
								),
						),
				),
		)
	tree1 := kdtree.NewTestKDTree(2, treeNodes)

	expTreeNodes1 := kdtree.NewKDNode(Tensor2D{25, 50}).
		SetLeft(
			kdtree.NewKDNode(Tensor2D{20, 15}).
				SetRight(
					kdtree.NewKDNode(Tensor2D{3, 25}),
				),
		).
		SetRight(
			kdtree.NewKDNode(Tensor2D{28, 47}).
				SetLeft(
					kdtree.NewKDNode(Tensor2D{30, 40}),
				).
				SetRight(
					kdtree.NewKDNode(Tensor2D{40, 60}),
				),
		)
	testTable := []struct {
		input    *kdtree.KDTree[Tensor2D]
		expected *kdtree.KDTree[Tensor2D]
	}{
		{
			input:    tree1,
			expected: kdtree.NewTestKDTree(2, expTreeNodes1),
		},
	}
	for _, v := range testTable {
		v.input.Balance()
		if !kdtree.IdenticalTrees(v.input, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, v.input)
		}
	}
}

func Test2DTree_Query(t *testing.T) {
	const dimensions = 2
	inputTensor2D := []Tensor2D{{1, 0}, {1, 8}, {2, 2}, {2, 10}, {3, 4}, {4, 1}, {5, 4}, {6, 8}, {7, 4}, {7, 7}, {8, 2}, {8, 5}, {9, 9}, {3, 6}, {4, 2}, {9, 2}, {6, 5}, {3, 8}, {6, 2}, {1, 3}, {3, 3}, {6, 4}, {9, 8}, {2, 1}, {2, 8}, {3, 1}, {7, 3}, {3, 9}, {4, 4}, {5, 3}, {9, 6}}
	tests := []struct {
		name     string
		tree     *kdtree.KDTree[Tensor2D]
		input    kdtree.RangeFunc[Tensor2D]
		expected []Tensor2D
	}{
		{
			name: "out of range x (lower)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td Tensor2D, i int) kdtree.RelativePosition {
				switch i {
				case -1:
					if x, y := td[0], td[1]; -2 <= x && x < -1 && 2 <= y && y < 10 {
						return kdtree.InRange
					}
					return kdtree.AfterRange
				case 0:
					if x := td[0]; x < -2 {
						return kdtree.BeforeRange
					} else if x >= -1 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				case 1:
					if y := td[1]; y < 2 {
						return kdtree.BeforeRange
					} else if y >= 10 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				}
				return kdtree.AfterRange
			},
			expected: []Tensor2D(nil),
		},
		{
			name: "out of range y (lower)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td Tensor2D, i int) kdtree.RelativePosition {
				switch i {
				case -1:
					if x, y := td[0], td[1]; 2 <= x && x < 10 && -2 <= y && y < -1 {
						return kdtree.InRange
					}
					return kdtree.AfterRange
				case 0:
					if x := td[0]; x < 2 {
						return kdtree.BeforeRange
					} else if x >= 10 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				case 1:
					if y := td[1]; y < -2 {
						return kdtree.BeforeRange
					} else if y >= -1 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				}
				return kdtree.AfterRange
			},
			expected: []Tensor2D(nil),
		},
		{
			name: "out of range x (higher)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td Tensor2D, i int) kdtree.RelativePosition {
				switch i {
				case -1:
					if x, y := td[0], td[1]; 20 <= x && x < 30 && 2 <= y && y < 10 {
						return kdtree.InRange
					}
					return kdtree.AfterRange
				case 0:
					if x := td[0]; x < 20 {
						return kdtree.BeforeRange
					} else if x >= 30 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				case 1:
					if y := td[1]; y < 2 {
						return kdtree.BeforeRange
					} else if y >= 10 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				}
				return kdtree.AfterRange
			},
			expected: []Tensor2D(nil),
		},
		{
			name: "out of range y (higher)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td Tensor2D, i int) kdtree.RelativePosition {
				switch i {
				case -1:
					if x, y := td[0], td[1]; 2 <= x && x < 10 && 20 <= y && y < 30 {
						return kdtree.InRange
					}
					return kdtree.AfterRange
				case 0:
					if x := td[0]; x < 2 {
						return kdtree.BeforeRange
					} else if x >= 10 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				case 1:
					if y := td[1]; y < 20 {
						return kdtree.BeforeRange
					} else if y >= 30 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				}
				return kdtree.AfterRange
			},
			expected: []Tensor2D(nil),
		},
		{
			name: "some values in range",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td Tensor2D, i int) kdtree.RelativePosition {
				switch i {
				case -1:
					if x, y := td[0], td[1]; 1 <= x && x < 2 && 2 <= y && y < 10 {
						return kdtree.InRange
					}
					return kdtree.AfterRange
				case 0:
					if x := td[0]; x < -2 {
						return kdtree.BeforeRange
					} else if x >= -1 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				case 1:
					if y := td[1]; y < 2 {
						return kdtree.BeforeRange
					} else if y >= 10 {
						return kdtree.AfterRange
					} else {
						return kdtree.InRange
					}
				}
				return kdtree.AfterRange
			},
			expected: []Tensor2D{{1, 3}, {1, 8}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.expected, test.tree.Query(test.input))
		})
	}
}
