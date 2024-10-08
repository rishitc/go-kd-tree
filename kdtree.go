package kdtree

import (
	"fmt"
	"sort"
	"strings"

	flatbuffers "github.com/google/flatbuffers/go"
	encoding "github.com/rishitc/go-kd-tree/internal/KDTreeEncoding"
	internal "github.com/rishitc/go-kd-tree/internal/utils"
)

func NewKDNode[T Comparable[T]](value T) *kdNode[T] {
	return &kdNode[T]{
		value: value,
	}
}

func NewKDTreeWithValues[T Comparable[T]](d int, vs []T) *KDTree[T] {
	size := len(vs)
	initialIndices := make([][]int, d)
	for cd := range initialIndices {
		initialIndices[cd] = internal.IotaSlice(len(vs))
		sort.Slice(initialIndices[cd], func(i, j int) bool {
			return vs[initialIndices[cd][i]].Order(vs[initialIndices[cd][j]], cd) < 0
		})
	}
	root := insertAllNew[T](vs, initialIndices, 0)
	return &KDTree[T]{
		dimensions: d,
		root:       root,
		isSetup:    true,
		size:       size,
	}
}

func NewKDTreeFromBytes[T Comparable[T]](encodedBytes []byte, decodeItemFunc func([]byte) T) *KDTree[T] {
	tree := encoding.GetRootAsKDTree(encodedBytes, 0)
	if encodingVersion != tree.VersionNumber() {
		panic("Unsupported encoding version number!")
	}
	itemsLength := tree.ItemsLength()
	if itemsLength != tree.InorderIndicesLength() {
		msg := fmt.Sprintf("The number of the indices (%d) are not the same as the number of items(%d)!",
			tree.InorderIndicesLength(), itemsLength)
		panic(msg)
	}
	// Note: This will be useful when I need to reconstruct the exact tree again.
	// For now the reconstructed tree will not be exactly the same. It will be a rebalanced tree.
	// This should not introduce any bugs in the code but rather make the tree based operations faster, when
	// loaded from binary.
	// preorderIndices := iotaSlice(itemsLength)
	// inorderIndices := make([]int, itemsLength)
	// inorderIndexLookup := make([]int, itemsLength)
	// for i := range inorderIndices {
	// 	idx := int(tree.InorderIndices(i))
	// 	inorderIndices[i] = idx
	// 	inorderIndexLookup[idx] = i
	// }
	items := make([]T, itemsLength)
	for i := 0; i < itemsLength; i++ {
		itemPtr := new(encoding.Item)
		if tree.Items(itemPtr, i) {
			item := decodeItemFunc(itemPtr.DataBytes())
			items[i] = item
		}
	}
	dimensions := int(tree.Dimensions())
	return NewKDTreeWithValues(dimensions, items)
}

func (t *KDTree[T]) FindMin(targetDimension int) (T, bool) {
	if t.root == nil || targetDimension >= t.dimensions {
		return t.zeroVal, false
	}
	res := findMin(t.dimensions, targetDimension, 0, t.root)
	if res == nil {
		return t.zeroVal, false
	}
	return *res, true
}

func (t *KDTree[T]) FindMax(targetDimension int) (T, bool) {
	if t.root == nil || targetDimension >= t.dimensions {
		return t.zeroVal, false
	}
	res := findMax(t.dimensions, targetDimension, 0, t.root)
	if res == nil {
		return t.zeroVal, false
	}
	return *res, true
}

func (t *KDTree[T]) NearestNeighbor(value T) (T, bool) {
	res := nearestNeighbor(t.dimensions, &value, nil, 0, t.root)
	if res == nil {
		return t.zeroVal, false
	}
	return *res, true
}

func (t *KDTree[T]) RangeSearch(getRelativePosition RangeFunc[T]) []T {
	var res []T
	rangeSearch(getRelativePosition, t.dimensions, &res, t.root, 0)
	return res
}

func (t *KDTree[T]) Values() []T {
	res := make([]T, 0, t.size)
	valuesImpl(t.root, &res)
	return res
}

func (t *KDTree[T]) Insert(value T) {
	if t.root == nil {
		t.root = NewKDNode(value)
		return
	}
	if insert(t.dimensions, value, 0, t.root) {
		t.size++
	}
}

func (t *KDTree[T]) Remove(value T) bool {
	ok := false
	t.root, ok = removeNode(t.dimensions, value, 0, t.root)
	if ok {
		t.size--
	}
	return ok
}

func valuesImpl[T Comparable[T]](r *kdNode[T], res *[]T) {
	if r == nil {
		return
	}

	*res = append(*res, r.value)
	valuesImpl(r.left, res)
	valuesImpl(r.right, res)
}

func (t *KDTree[T]) String() string {
	b := strings.Builder{}
	var q Queue[*kdNode[T]] = NewLLQueue[*kdNode[T]]()
	q.Push(t.root)
	for !q.Empty() {
		size := q.Size()
		for i := 0; i < size; i++ {
			n, _ := q.Pop()
			if n != nil {
				b.WriteString(n.value.String())
				b.WriteString(", ")
				q.Push(n.left)
				q.Push(n.right)
			} else {
				b.WriteString("nil, ")
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

// Implementation inspired by: https://eli.thegreenplace.net/2009/11/23/visualizing-binary-trees-with-graphviz
func (t *KDTree[T]) Dot() string {
	b := strings.Builder{}
	b.WriteString("digraph BST {\n")

	node := t.root
	if node == nil {
		b.WriteString("\n")
	} else {
		nodeCount := 0
		currentNode := fmt.Sprintf("node%d", nodeCount)
		nodeCount++
		currNodeDef := fmt.Sprintf("    %s [label=\"%s\"]\n", currentNode, node.value.String())
		b.WriteString(currNodeDef)
		dot(node, &b, &nodeCount, currentNode)
	}

	b.WriteString("}\n")
	return b.String()
}

func dot[T Comparable[T]](node *kdNode[T], b *strings.Builder, nodeCount *int, currentNode string) {
	leftNode := fmt.Sprintf("node%d", *nodeCount)
	*nodeCount++
	if node.left != nil {
		leftNodeDef := fmt.Sprintf("    %s [label=\"%s\"];\n", leftNode, node.left.value.String())
		b.WriteString(leftNodeDef)
		b.WriteString(fmt.Sprintf("    %s -> %s;\n", currentNode, leftNode))
		dot(node.left, b, nodeCount, leftNode)
	} else {
		leftNodeDef := fmt.Sprintf("    %s [shape=point];\n", leftNode)
		b.WriteString(leftNodeDef)
		b.WriteString(fmt.Sprintf("    %s -> %s;\n", currentNode, leftNode))
	}

	rightNode := fmt.Sprintf("node%d", *nodeCount)
	*nodeCount++
	if node.right != nil {
		rightNodeDef := fmt.Sprintf("    %s [label=\"%s\"];\n", rightNode, node.right.value.String())
		b.WriteString(rightNodeDef)
		b.WriteString(fmt.Sprintf("    %s -> %s;\n", currentNode, rightNode))
		dot(node.right, b, nodeCount, rightNode)
	} else {
		rightNodeDef := fmt.Sprintf("    %s [shape=point];\n", rightNode)
		b.WriteString(rightNodeDef)
		b.WriteString(fmt.Sprintf("    %s -> %s;\n", currentNode, rightNode))
	}
}

const encodingVersion uint32 = 0

func (t *KDTree[T]) Encode() []byte {
	encodedPreorderItems := preorderTraversal(t.root)
	itemCount := len(encodedPreorderItems)
	if itemCount != t.size {
		msg := fmt.Sprintf("itemCount (%d) and t.size (%d) don't have the same size! Some bookkeeping has gone wrong!", itemCount, t.size)
		panic(msg)
	}
	encodedInorderIndices := inorderTraversal(t.root, t.size)

	builder := flatbuffers.NewBuilder(256)

	encoding.KDTreeStartInorderIndicesVector(builder, itemCount)
	for i := itemCount - 1; i >= 0; i-- {
		idx := encodedInorderIndices[i]
		builder.PrependInt64(int64(idx))
	}
	inorderIndices := builder.EndVector(itemCount)

	var encodedItems []flatbuffers.UOffsetT
	for i := 0; i < itemCount; i++ {
		item := encodedPreorderItems[i]
		size := len(item)

		encoding.ItemStartDataVector(builder, size)
		for i := size - 1; i >= 0; i-- {
			itemByte := item[i]
			builder.PrependByte(itemByte)
		}
		itemBytesVector := builder.EndVector(size)

		encoding.ItemStart(builder)
		encoding.ItemAddData(builder, itemBytesVector)
		encodedItem := encoding.ItemEnd(builder)

		encodedItems = append(encodedItems, encodedItem)
	}

	encoding.KDTreeStartItemsVector(builder, itemCount)
	for i := itemCount - 1; i >= 0; i-- {
		builder.PrependUOffsetT(encodedItems[i])
	}
	items := builder.EndVector(itemCount)

	encoding.KDTreeStart(builder)
	encoding.KDTreeAddVersionNumber(builder, encodingVersion)
	encoding.KDTreeAddDimensions(builder, uint32(t.dimensions))
	encoding.KDTreeAddInorderIndices(builder, inorderIndices)
	encoding.KDTreeAddItems(builder, items)
	encodedKDTree := encoding.KDTreeEnd(builder)
	builder.Finish(encodedKDTree)
	return builder.FinishedBytes()
}

// Balance rebalance the k-d tree by recreating it.
func (t *KDTree[T]) Balance() {
	t.root = NewKDTreeWithValues(t.dimensions, t.Values()).root
}

func rangeSearch[T Comparable[T]](getRelativePosition RangeFunc[T], d int, res *[]T, r *kdNode[T], cd int) {
	if r == nil {
		return
	}

	rel := getRelativePosition(r.value, -1)
	if rel == InRange {
		*res = append(*res, r.value)
	}

	ncd := (cd + 1) % d
	switch relInCD := getRelativePosition(r.value, cd); relInCD {
	case BeforeRange:
		rangeSearch(getRelativePosition, d, res, r.right, ncd)
	case AfterRange:
		rangeSearch(getRelativePosition, d, res, r.left, ncd)
	case InRange:
		rangeSearch(getRelativePosition, d, res, r.left, ncd)
		rangeSearch(getRelativePosition, d, res, r.right, ncd)
	default:
		panic(fmt.Sprintf("Invalid value returned: %v", relInCD))
	}
}

func preorderTraversal[T Comparable[T]](r *kdNode[T]) [][]byte {
	var res [][]byte
	preorderTraversalImpl(r, &res)
	return res
}

func preorderTraversalImpl[T Comparable[T]](r *kdNode[T], res *[][]byte) {
	if r == nil {
		return
	}
	*res = append(*res, r.value.Encode())
	preorderTraversalImpl(r.left, res)
	preorderTraversalImpl(r.right, res)
}

func inorderTraversal[T Comparable[T]](r *kdNode[T], size int) []int {
	preorderIndex := 0
	inorderIndex := 0
	res := make([]int, size)
	inorderTraversalImpl(r, &preorderIndex, &inorderIndex, &res)
	return res
}

func inorderTraversalImpl[T Comparable[T]](r *kdNode[T], preorderIndex, inorderIndex *int, res *[]int) {
	if r == nil {
		return
	}
	currPreorderIndex := *preorderIndex
	*preorderIndex++
	inorderTraversalImpl(r.left, preorderIndex, inorderIndex, res)
	currInorderIndex := *inorderIndex
	*inorderIndex++
	(*res)[currInorderIndex] = currPreorderIndex
	inorderTraversalImpl(r.right, preorderIndex, inorderIndex, res)
}

func insertAllNew[T Comparable[T]](vs []T, initialIndices [][]int, cd int) *kdNode[T] {
	if len(initialIndices[0]) == 0 {
		return nil
	}
	dims := len(initialIndices)
	cutIndex := initialIndices[0]
	mv, mvIdx := midValue(vs, cutIndex)
	n := NewKDNode(mv)

	// Split initialIndices
	temp := make([]int, len(cutIndex))
	copy(temp, cutIndex)

	lh := make([][]int, dims)
	uh := make([][]int, dims)
	si := (len(initialIndices[0]) - 1) / 2
	for i := 0; i < dims; i++ {
		indexArray := initialIndices[i]
		lh[i] = indexArray[:si]
		uh[i] = indexArray[si+1:]
	}

	for i := 1; i < dims; i++ {
		lhi := 0
		uhi := 0
		indexArray := initialIndices[i]
		for _, idx := range indexArray {
			if idx == mvIdx {
				continue
			}
			v := vs[idx]
			if v.Order(mv, cd) < 0 {
				lh[i-1][lhi] = idx
				lhi++
			} else {
				uh[i-1][uhi] = idx
				uhi++
			}
		}
	}
	copy(initialIndices[dims-1], temp)

	ncd := (cd + 1) % dims
	n.left = insertAllNew(vs, lh, ncd)
	n.right = insertAllNew(vs, uh, ncd)
	return n
}

func removeNode[T Comparable[T]](d int, value T, cd int, r *kdNode[T]) (*kdNode[T], bool) {
	if r == nil {
		return nil, false
	}
	ncd := (cd + 1) % d
	ok := false
	if r.value.Dist(value) == 0 {
		ok = true
		if r.right != nil {
			r.value = *findMin(d, cd, ncd, r.right)
			r.right, ok = removeNode(d, r.value, ncd, r.right)
		} else if r.left != nil {
			r.value = *findMin(d, cd, ncd, r.left)
			r.right, ok = removeNode(d, r.value, ncd, r.left)
			r.left = nil
		} else {
			r = nil
		}
	} else if value.Order(r.value, cd) < 0 {
		r.left, ok = removeNode(d, value, ncd, r.left)
	} else {
		r.right, ok = removeNode(d, value, ncd, r.right)
	}
	return r, ok
}

func insert[T Comparable[T]](d int, value T, cd int, r *kdNode[T]) bool {
	for value.Dist(r.value) != 0 {
		rel := value.Order(r.value, cd)
		if rel < 0 {
			if r.left == nil {
				r.left = NewKDNode(value)
				return true
			}
			r = r.left
		} else {
			if r.right == nil {
				r.right = NewKDNode(value)
				return true
			}
			r = r.right
		}
		cd = (cd + 1) % d
	}
	return false
}

func nearestNeighbor[T Comparable[T]](d int, v, nn *T, cd int, r *kdNode[T]) *T {
	if r == nil {
		return nil
	}

	var nextBranch, otherBranch *kdNode[T]
	if (*v).Order(r.value, cd) < 0 /* [cd] < r.value[cd]*/ {
		nextBranch, otherBranch = r.left, r.right
	} else {
		nextBranch, otherBranch = r.right, r.left
	}
	ncd := (cd + 1) % d
	nn = nearestNeighbor(d, v, nn, ncd, nextBranch)
	nn = closest(v, nn, &r.value)

	nearestDist := internal.Abs(distance(v, nn))
	dist := internal.Abs((*v).DistDim(r.value, cd))

	if dist <= nearestDist {
		nn = closest(v, nearestNeighbor(d, v, nn, ncd, otherBranch), nn)
	}

	return nn
}

func (t *KDTree[T]) KNN(value T, k int) []T {
	if t == nil || t.root == nil || t.size < k {
		return nil
	}

	pqRes := NewBoundedPriorityQueue[T](k)
	knn(k, t.dimensions, &value, &pqRes, 0, t.root)

	res := make([]T, 0, k)
	for range k {
		// heap with a preset capacity
		d := *internal.Pop(&pqRes).Data
		res = append(res, d)
	}

	return res
}

type direction bool

const (
	left  direction = true
	right           = false
)

type nodeInfo[T Comparable[T]] struct {
	node *kdNode[T]
	dir  direction
}

func knn[T Comparable[T]](k, d int, v *T, pq *BoundedPriorityQueue[T], cd int, r *kdNode[T]) {
	if r == nil {
		return
	}

	ncd := cd

	var path []nodeInfo[T]
	for r != nil {
		info := nodeInfo[T]{
			node: r,
		}
		if rel := (*v).Order(r.value, ncd); rel < 0 {
			r = r.left
			info.dir = left
		} else {
			r = r.right
			info.dir = right
		}
		path = append(path, info)

		ncd = (ncd + 1) % d
	}

	ncd = (ncd - 1 + d) % d // Go back to the dimension used for splitting at the leaf node.
	for path, cn, cDir := popLast(path); cn != nil; path, cn, cDir = popLast(path) {
		currentDistance := (*v).Dist(cn.value)
		internal.Push(pq, Item[T]{
			Data:     &cn.value,
			Priority: currentDistance,
		})

		if pq.Len() < pq.Capacity() || (*v).DistDim(cn.value, ncd) < getFarthestDistance(pq) {
			var next *kdNode[T]
			if cDir == left {
				next = cn.right
			} else {
				next = cn.left
			}
			knn(k, d, v, pq, (ncd+1)%d, next)
		}
		ncd = (ncd - 1 + d) % d
	}
}

func getFarthestDistance[T Comparable[T]](pq *BoundedPriorityQueue[T]) int {
	v := pq.Peek()
	return v.Priority
}

func popLast[T Comparable[T]](arr []nodeInfo[T]) ([]nodeInfo[T], *kdNode[T], direction) {
	if len(arr) == 0 {
		return arr, nil, left
	}
	li := len(arr) - 1
	return arr[:li], arr[li].node, arr[li].dir
}

func findMin[T Comparable[T]](d, tcd, cd int, r *kdNode[T]) *T {
	if r == nil {
		return nil
	}

	var lMin *T
	var rMin *T
	ncd := (cd + 1) % d
	lMin = findMin(d, tcd, ncd, r.left)
	if tcd != cd {
		rMin = findMin(d, tcd, ncd, r.right)
	}
	if lMin == nil && rMin == nil {
		return &r.value
	} else if lMin == nil {
		if (*rMin).Order(r.value, tcd) < 0 {
			return rMin
		}
		return &r.value
	} else if rMin == nil {
		if (*lMin).Order(r.value, tcd) < 0 {
			return lMin
		}
		return &r.value
	} else {
		// temp := []*T{lMin, rMin, &r.value}
		// sort.Slice(temp, func(i, j int) bool {
		// 	return (*temp[i]).Order(*temp[j], tcd) < 0
		// })
		// return temp[0]
		return min(lMin, min(rMin, &r.value, tcd), tcd)
	}
}

func findMax[T Comparable[T]](d, tcd, cd int, r *kdNode[T]) *T {
	if r == nil {
		return nil
	}

	var lMax *T
	var rMax *T
	ncd := (cd + 1) % d
	rMax = findMax(d, tcd, ncd, r.right)
	if tcd != cd {
		lMax = findMax(d, tcd, ncd, r.left)
	}
	if lMax == nil && rMax == nil {
		return &r.value
	} else if lMax == nil {
		if (*rMax).Order(r.value, tcd) > 0 {
			return rMax
		}
		return &r.value
	} else if rMax == nil {
		if (*lMax).Order(r.value, tcd) > 0 {
			return lMax
		}
		return &r.value
	} else {
		return max(lMax, max(rMax, &r.value, tcd), tcd)
	}
}
