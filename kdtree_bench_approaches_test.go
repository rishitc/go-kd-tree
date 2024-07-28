package kdtree

import "sort"

func OldKDTreeWithValues[T Comparable[T]](d int, vs []T) *KDTree[T] {
	root := insertAllOld(d, vs, 0)
	return &KDTree[T]{
		dimensions: d,
		root:       root,
		isSetup:    true,
	}
}

func insertAllOld[T Comparable[T]](d int, vs []T, cd int) *kdNode[T] {
	if len(vs) == 0 {
		return nil
	}
	sort.Slice(vs, func(i, j int) bool {
		return vs[i].Order(vs[j], cd) == Lesser // vs[i].Get(cd) < vs[j].Get(cd)
	})
	mi := (len(vs) - 1) / 2
	mv := vs[mi]
	n := NewKDNode(mv)

	ncd := (cd + 1) % d

	lv := vs[:mi]
	n.left = insertAllOld(d, lv, ncd)

	rv := vs[mi+1:]
	n.right = insertAllOld(d, rv, ncd)
	return n
}

func (t *KDTree[T]) OldValues() []T {
	var res []T
	oldValuesImpl(t.root, &res)
	return res
}

func oldValuesImpl[T Comparable[T]](r *kdNode[T], res *[]T) {
	if r == nil {
		return
	}

	*res = append(*res, r.value)
	valuesImpl(r.left, res)
	valuesImpl(r.right, res)
}
