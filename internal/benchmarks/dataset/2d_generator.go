//go:build trace

package dataset

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

func Trace2DGenerator(tracePath string) {
	// Open the file for writing
	file, err := os.Create(tracePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Create a CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Create a map to store unique pairs
	uniquePairs := make(map[string]struct{})

	// Generate unique pairs
	for len(uniquePairs) < 1000000 {
		x := rand.Intn(1000000)
		y := rand.Intn(1000000)
		pair := fmt.Sprintf("%d,%d", x, y)

		if _, exists := uniquePairs[pair]; !exists {
			uniquePairs[pair] = struct{}{}
			err := writer.Write([]string{strconv.Itoa(x), strconv.Itoa(y)})
			if err != nil {
				panic(err)
			}
		}
	}
}
