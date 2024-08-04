package kdtree

import (
	"testing"

	types "github.com/rishitc/go-kd-tree/internal/types"
)

func Test2DDeleteAllNodesInTree(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{5, 6})
	tree := NewTestKDTree(2, treeNodes)

	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{5, 6},
			expected: NewTestKDTree[types.Tensor2D](2, nil),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=YhQLGxiH7BbG9D8s&t=37
func Test2DDeleteLeafNode(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}),
		).
		SetRight(
			NewKDNode(types.Tensor2D{40, 60}).
				SetLeft(
					NewKDNode(types.Tensor2D{30, 40}),
				),
		)
	tree := NewTestKDTree(2, treeNodes)

	expTreeNodes := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}),
		).
		SetRight(
			NewKDNode(types.Tensor2D{40, 60}),
		)
	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{30, 40},
			expected: NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=-tQZZtNASyMXnhNc&t=90
func Test2DDeleteNodeWithRightSubtree(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}).
				SetLeft(
					NewKDNode(types.Tensor2D{20, 15}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{40, 60}).
				SetLeft(
					NewKDNode(types.Tensor2D{30, 40}).
						SetLeft(
							NewKDNode(types.Tensor2D{28, 17}),
						),
				),
		)
	tree := NewTestKDTree(2, treeNodes)

	expTreeNodes := NewKDNode(types.Tensor2D{28, 17}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}).
				SetLeft(
					NewKDNode(types.Tensor2D{20, 15}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{40, 60}).
				SetLeft(
					NewKDNode(types.Tensor2D{30, 40}),
				),
		)
	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{25, 50},
			expected: NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

// https://youtu.be/DkBNF98MV1Q?si=v-TuZNV9YiTmCOFg&t=189
func Test2DDeleteNodeWithLeftSubtreeOnly(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}).
				SetLeft(
					NewKDNode(types.Tensor2D{20, 15}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{40, 60}).
				SetLeft(
					NewKDNode(types.Tensor2D{30, 40}).
						SetLeft(
							NewKDNode(types.Tensor2D{28, 47}),
						),
				),
		)
	tree := NewTestKDTree(2, treeNodes)

	expTreeNodes := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}).
				SetLeft(
					NewKDNode(types.Tensor2D{20, 15}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{30, 40}).
				SetRight(
					NewKDNode(types.Tensor2D{28, 47}),
				),
		)
	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{40, 60},
			expected: NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

func Test2DDeleteNode1(t *testing.T) {
	const dimensions = 2
	ps := []types.Tensor2D{
		{5, 6},
		{4, 10},
		{4, 20},
	}
	tree := NewKDTreeWithValues(dimensions, ps)
	expTree := NewKDNode(types.Tensor2D{4, 20}).SetLeft(NewKDNode(types.Tensor2D{4, 10}))

	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{5, 6},
			expected: NewTestKDTree(2, expTree),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}
func Test2DDeleteNode2(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{5, 6}).
		SetLeft(
			NewKDNode(types.Tensor2D{4, 10}).
				SetRight(
					NewKDNode(types.Tensor2D{4, 20}),
				),
		)
	tree := NewTestKDTree(2, treeNodes)

	expTreeNodes := NewKDNode(types.Tensor2D{4, 10}).
		SetRight(
			NewKDNode(types.Tensor2D{4, 20}),
		)
	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{5, 6},
			expected: NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
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
	treeNodes := NewKDNode(types.Tensor2D{35, 60}).
		SetLeft(
			NewKDNode(types.Tensor2D{20, 45}).
				SetLeft(
					NewKDNode(types.Tensor2D{10, 35}).
						SetRight(
							NewKDNode(types.Tensor2D{20, 20}),
						),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{60, 80}).
				SetLeft(
					NewKDNode(types.Tensor2D{80, 40}).
						SetLeft(
							NewKDNode(types.Tensor2D{50, 30}).
								SetLeft(
									NewKDNode(types.Tensor2D{70, 20}).
										SetLeft(
											NewKDNode(types.Tensor2D{60, 10}),
										),
								),
						).
						SetRight(
							NewKDNode(types.Tensor2D{90, 60}),
						),
				),
		)
	tree := NewTestKDTree(2, treeNodes)

	expTreeNodes := NewKDNode(types.Tensor2D{50, 30}).
		SetLeft(
			NewKDNode(types.Tensor2D{20, 45}).
				SetLeft(
					NewKDNode(types.Tensor2D{10, 35}).
						SetRight(
							NewKDNode(types.Tensor2D{20, 20}),
						),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{60, 80}).
				SetLeft(
					NewKDNode(types.Tensor2D{80, 40}).
						SetLeft(
							NewKDNode(types.Tensor2D{60, 10}).
								SetRight(
									NewKDNode(types.Tensor2D{70, 20}),
								),
						).
						SetRight(
							NewKDNode(types.Tensor2D{90, 60}),
						),
				),
		)
	testTable := []struct {
		input    types.Tensor2D
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    types.Tensor2D{35, 60},
			expected: NewTestKDTree(2, expTreeNodes),
		},
	}
	for _, v := range testTable {
		ok := tree.Delete(v.input)
		if !ok || !IdenticalTrees(tree, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, tree)
		}
	}
}

func Test2DTreeEncodeDecode(t *testing.T) {
	const dimensions = 2
	treeNodes := NewKDNode(types.Tensor2D{4, 4}).
		SetLeft(
			NewKDNode(types.Tensor2D{2, 2}).
				SetLeft(
					NewKDNode(types.Tensor2D{1, 1}),
				).
				SetRight(
					NewKDNode(types.Tensor2D{3, 2}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{6, 1}).
				SetLeft(
					NewKDNode(types.Tensor2D{9, 0}),
				).
				SetRight(
					NewKDNode(types.Tensor2D{5, 8}).
						SetRight(
							NewKDNode(types.Tensor2D{8, 7}),
						),
				),
		)
	expectedTree := NewTestKDTree(dimensions, treeNodes)
	encodedTreeBytes := expectedTree.Encode()
	tree := NewKDTreeFromBytes(encodedTreeBytes, types.DecodeTensor2D)
	if !IdenticalTrees(tree, expectedTree) {
		t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", expectedTree, tree)
	}
}

func Test2DTreeBalance(t *testing.T) {
	treeNodes := NewKDNode(types.Tensor2D{20, 15}).
		SetLeft(
			NewKDNode(types.Tensor2D{3, 25}),
		).
		SetRight(
			NewKDNode(types.Tensor2D{30, 40}).
				SetRight(
					NewKDNode(types.Tensor2D{25, 50}).
						SetRight(
							NewKDNode(types.Tensor2D{28, 47}).
								SetRight(
									NewKDNode(types.Tensor2D{40, 60}),
								),
						),
				),
		)
	tree1 := NewTestKDTree(2, treeNodes)

	expTreeNodes1 := NewKDNode(types.Tensor2D{25, 50}).
		SetLeft(
			NewKDNode(types.Tensor2D{20, 15}).
				SetRight(
					NewKDNode(types.Tensor2D{3, 25}),
				),
		).
		SetRight(
			NewKDNode(types.Tensor2D{28, 47}).
				SetLeft(
					NewKDNode(types.Tensor2D{30, 40}),
				).
				SetRight(
					NewKDNode(types.Tensor2D{40, 60}),
				),
		)
	testTable := []struct {
		input    *KDTree[types.Tensor2D]
		expected *KDTree[types.Tensor2D]
	}{
		{
			input:    tree1,
			expected: NewTestKDTree(2, expTreeNodes1),
		},
	}
	for _, v := range testTable {
		v.input.Balance()
		if !IdenticalTrees(v.input, v.expected) {
			t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", v.expected, v.input)
		}
	}
}
