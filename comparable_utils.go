package kdtree

func min[T Comparable[T]](lhs, rhs *T, tcd int) *T {
	if (*lhs).Order(*rhs, tcd) == Lesser {
		return lhs
	}
	return rhs
}

func max[T Comparable[T]](lhs, rhs *T, tcd int) *T {
	if (*lhs).Order(*rhs, tcd) == Greater {
		return lhs
	}
	return rhs
}

func distance[T Comparable[T]](src, dst *T) int {
	return (*src).Dist(*dst)
}

func closest[T Comparable[T]](v, nn1, nn2 *T) *T {
	if nn1 == nil && nn2 == nil {
		panic("Both `nn1` and `nn2` inputs are nil!")
	}

	if nn1 == nil {
		return nn2
	}
	if nn2 == nil {
		return nn1
	}
	if distance(v, nn1) < distance(v, nn2) {
		return nn1
	}
	return nn2
}

func midValue[T Comparable[T]](vs []T, cutIndex []int) (T, int) {
	i := (len(cutIndex) - 1) / 2
	mvi := cutIndex[i]
	return vs[mvi], mvi
}
