package kyro_test

import (
	"fmt"
)

type Point2D [2]float64

func (p *Point2D) Dimensions() int {
	return 2
}

func (p *Point2D) Dimension(i int) float64 {
	if i == 0 {
		return p[0]
	}
	return p[1]
}

func (p *Point2D) String() string {
	return fmt.Sprintf("{%.2f %.2f}", p[0], p[1])
}
