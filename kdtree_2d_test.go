package kdtree_test

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
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
