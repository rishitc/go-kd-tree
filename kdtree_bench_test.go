package kdtree_test

import (
	"testing"

	kdtree "github.com/rishitc/go-kd-tree"
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

func BenchmarkOldKDTreeValues(b *testing.B) {
	const numArrays = 10000 // number of arrays
	const dimensions = 3

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Generate a slice of random arrays
		arrays := generateRandomArrays(numArrays)
		// Run the KD-tree construction
		t := kdtree.NewKDTreeWithValues(dimensions, arrays)
		b.StartTimer()
		// Get the values
		t.OldValues()
	}
}

func BenchmarkNewKDTreeValues(b *testing.B) {
	const numArrays = 10000 // number of arrays
	const dimensions = 3

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Generate a slice of random arrays
		arrays := generateRandomArrays(numArrays)
		// Run the KD-tree construction
		t := kdtree.NewKDTreeWithValues(dimensions, arrays)
		b.StartTimer()
		// Get the values
		t.Values()
	}
}
