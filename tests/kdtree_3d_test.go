package tests

import (
	"slices"
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
	types "github.com/rishitc/go-kd-tree/internal/types"
)

func Test3DNearestNeighbor1(t *testing.T) {
	const dimensions = 3
	ps := []types.Tensor3D{
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
		input, expected types.Tensor3D
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
	ps := []types.Tensor3D{
		{5, 4},
		{3, 1},
		{2, 6},
		{8, 7},
		{10, 2},
		{13, 3},
	}
	tree := kdtree.NewKDTreeWithValues(dimensions, ps)
	testTable := []struct {
		input, expected types.Tensor3D
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
	ps := []types.Tensor3D{
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
		input, expected types.Tensor3D
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
	ps := []types.Tensor3D{
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
		input, expected types.Tensor3D
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
