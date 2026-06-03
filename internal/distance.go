package internal

// EuclideanDistSq calcula a distância euclidiana ao quadrado (evita o custo de sqrt).
func EuclideanDistSq(a, b *[VectorDimsPad]float32) float32 {
	var sum float32
	for i := 0; i < VectorDimsPad; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
