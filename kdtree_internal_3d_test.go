package kdtree

import (
	"testing"

	types "github.com/rishitc/go-kd-tree/internal/types"
)

func Test3DTreeEncodeDecode(t *testing.T) {
	const dimensions = 3
	treeNodes := NewKDNode(types.Tensor3D{7, 77, 10}).
		SetLeft(
			NewKDNode(types.Tensor3D{0, 5, 1}).
				SetLeft(
					NewKDNode(types.Tensor3D{1, 2, 3}),
				).
				SetRight(
					NewKDNode(types.Tensor3D{0, 50, 1}).
						SetRight(
							NewKDNode(types.Tensor3D{1, 7, 1}),
						),
				),
		).
		SetRight(
			NewKDNode(types.Tensor3D{9, 16, 4}).
				SetLeft(
					NewKDNode(types.Tensor3D{90, 0, 1}).
						SetRight(
							NewKDNode(types.Tensor3D{44, 5, 14}),
						),
				).
				SetRight(
					NewKDNode(types.Tensor3D{10, 20, 1}).
						SetRight(
							NewKDNode(types.Tensor3D{31, 42, 49}),
						),
				),
		)
	expectedTree := NewTestKDTree(dimensions, treeNodes)
	encodedTreeBytes := expectedTree.Encode()
	tree := NewKDTreeFromBytes(encodedTreeBytes, types.DecodeTensor3D)
	if !IdenticalTrees(tree, expectedTree) {
		t.Fatalf("Tree does not match expected tree structure\nExpected:\n%s\nGot:\n%s", expectedTree, tree)
	}
}
