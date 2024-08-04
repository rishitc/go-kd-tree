package kdtree

func NewTestKDTree[T Comparable[T]](d int, r *kdNode[T]) *KDTree[T] {
	return &KDTree[T]{
		dimensions: d,
		root:       r,
		isSetup:    true,
		zeroVal:    *new(T),
		size:       countNodes(r),
	}
}

func countNodes[T Comparable[T]](r *kdNode[T]) int {
	if r == nil {
		return 0
	}
	return 1 + countNodes(r.left) + countNodes(r.right)
}

func (n *kdNode[T]) SetLeft(nn *kdNode[T]) *kdNode[T] {
	n.left = nn
	return n
}

func (n *kdNode[T]) SetRight(nn *kdNode[T]) *kdNode[T] {
	n.right = nn
	return n
}

func IdenticalTrees[T Comparable[T]](lhs, rhs *KDTree[T]) bool {
	stk := [][2]*kdNode[T]{{lhs.root, rhs.root}}
	for len(stk) != 0 {
		p := stk[len(stk)-1][0]
		q := stk[len(stk)-1][1]
		stk = stk[:len(stk)-1]
		if p != nil && q != nil && p.value.Dist(q.value) == 0 {
			stk = append(stk, [2]*kdNode[T]{p.left, q.left}, [2]*kdNode[T]{p.right, q.right})
		} else if p != nil || q != nil {
			return false
		}
	}
	return true
}
