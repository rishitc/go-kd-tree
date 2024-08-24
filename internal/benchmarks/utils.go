package benchmarks

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func Read2DTrace(filename string) ([][]int, error) {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	// Read all records from the CSV
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Create a slice to hold the parsed pairs
	var result [][]int

	// Parse the records into pairs of integers
	for _, record := range records {
		if len(record) != 2 {
			return nil, fmt.Errorf("unexpected record length: %v", record)
		}

		// Convert the strings to integers
		x, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, err
		}

		y, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, err
		}

		// Append the pair to the result slice
		result = append(result, []int{x, y})
	}

	return result, nil
}
