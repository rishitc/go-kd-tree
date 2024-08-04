package tests

import (
	"slices"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
	types "github.com/rishitc/go-kd-tree/internal/types"
	"github.com/stretchr/testify/assert"
)

func Test2DNearestNeighbor1(t *testing.T) {
	const dimensions = 2
	ps := []types.Tensor2D{
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
		input, expected types.Tensor2D
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
	ps := []types.Tensor2D{
		{5, 4},
		{3, 1},
		{2, 6},
		{8, 7},
		{10, 2},
		{13, 3},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected types.Tensor2D
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
	ps := []types.Tensor2D{
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
		input, expected types.Tensor2D
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
	ps := []types.Tensor2D{
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
		input, expected types.Tensor2D
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
	ps := []types.Tensor2D{
		{50, 50},
		{80, 40},
		{10, 60},
		{51, 38},
		{48, 38},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected types.Tensor2D
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

func tensor2DSortFunc(a, b types.Tensor2D) int {
	if a[0] != b[0] {
		return a[0] - b[0]
	} else {
		return a[1] - b[1]
	}
}

func Test2DKNearestNeighbor1(t *testing.T) {
	const dimensions = 2
	type input struct {
		p types.Tensor2D
		k int
	}
	ps := []types.Tensor2D{
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
		expected []types.Tensor2D
	}{
		"Find the 2 closest neighbors to a point that is not in the KD tree.": {
			input:    input{p: [2]int{25, 25}, k: 2},
			expected: []types.Tensor2D{{40, 20}, {10, 25}},
		},
		"The closest neighbor to a point that is in the KD tree.": {
			input:    input{p: [2]int{60, 90}, k: 1},
			expected: []types.Tensor2D{{60, 90}},
		},
		"The three closest neighbors to a point that is in the KD tree.": {
			input:    input{p: [2]int{70, 70}, k: 3},
			expected: []types.Tensor2D{{50, 50}, {60, 90}, {70, 70}},
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
	ps := []types.Tensor2D{
		{50, 50},
		{80, 40},
		{10, 60},
		{51, 38},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	v := types.Tensor2D{48, 38}
	tree.Add(v)
	testTable := []struct {
		input, expected types.Tensor2D
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
	tree := kdtree.NewKDTreeWithValues(dimensions, []types.Tensor2D{})
	ps := []types.Tensor2D{
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
		input, expected types.Tensor2D
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
	ps := []types.Tensor2D{
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
		expected types.Tensor2D
	}{
		{
			input:    0,
			expected: types.Tensor2D{1, 10},
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
	ps := []types.Tensor2D{
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
		expected types.Tensor2D
	}{
		{
			input:    1,
			expected: types.Tensor2D{55, 1},
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
	ps := []types.Tensor2D{
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
		expected types.Tensor2D
	}{
		{
			input:    0,
			expected: types.Tensor2D{70, 70},
		},
		{
			input:    1,
			expected: types.Tensor2D{35, 90},
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
	ps := []types.Tensor2D{
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
		expected types.Tensor2D
	}{
		{
			input:    1,
			expected: types.Tensor2D{55, 1},
		},
	}
	for _, v := range testTable {
		nn, ok := tree.FindMin(v.input)
		if !ok || !slices.Equal(nn[:], v.expected[:]) {
			t.Fatalf("Expected closest point: %v, got %v", v.expected, nn)
		}
	}
}

func Test2DTree_Query(t *testing.T) {
	const dimensions = 2
	inputTensor2D := []types.Tensor2D{{1, 0}, {1, 8}, {2, 2}, {2, 10}, {3, 4}, {4, 1}, {5, 4}, {6, 8}, {7, 4}, {7, 7}, {8, 2}, {8, 5}, {9, 9}, {3, 6}, {4, 2}, {9, 2}, {6, 5}, {3, 8}, {6, 2}, {1, 3}, {3, 3}, {6, 4}, {9, 8}, {2, 1}, {2, 8}, {3, 1}, {7, 3}, {3, 9}, {4, 4}, {5, 3}, {9, 6}}
	tests := []struct {
		name     string
		tree     *kdtree.KDTree[types.Tensor2D]
		input    kdtree.RangeFunc[types.Tensor2D]
		expected []types.Tensor2D
	}{
		{
			name: "out of range x (lower)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td types.Tensor2D, i int) kdtree.RelativePosition {
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
			expected: []types.Tensor2D(nil),
		},
		{
			name: "out of range y (lower)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td types.Tensor2D, i int) kdtree.RelativePosition {
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
			expected: []types.Tensor2D(nil),
		},
		{
			name: "out of range x (higher)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td types.Tensor2D, i int) kdtree.RelativePosition {
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
			expected: []types.Tensor2D(nil),
		},
		{
			name: "out of range y (higher)",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td types.Tensor2D, i int) kdtree.RelativePosition {
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
			expected: []types.Tensor2D(nil),
		},
		{
			name: "some values in range",
			tree: kdtree.NewKDTreeWithValues(dimensions, inputTensor2D),
			input: func(td types.Tensor2D, i int) kdtree.RelativePosition {
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
			expected: []types.Tensor2D{{1, 3}, {1, 8}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.expected, test.tree.Query(test.input))
		})
	}
}
