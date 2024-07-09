package kdtree_test

import (
	kdtree "kd-tree"
	"testing"
)

func BenchmarkNewKDTreeConstruction(b *testing.B) {
	const numArrays = 10000 // number of arrays
	const dimensions = 3

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Generate a slice of random arrays
		arrays := generateRandomArrays(numArrays)
		// Run the KD-tree construction
		b.StartTimer()
		kdtree.NewKDTreeWithValues(dimensions, arrays)
	}
}

func BenchmarkOldKDTreeConstruction(b *testing.B) {
	const numArrays = 10000 // number of arrays
	const dimensions = 3

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Generate a slice of random arrays
		arrays := generateRandomArrays(numArrays)
		// Run the KD-tree construction
		b.StartTimer()
		kdtree.OldKDTreeWithValues(dimensions, arrays)
	}
}
