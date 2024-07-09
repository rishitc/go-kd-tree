package kdtree

func iotaSlice(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}
