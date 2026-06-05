package internal

// EuclideanDistSq calcula a distância euclidiana ao quadrado entre duas referências quantizadas (uint8) usando aritmética inteira.
func EuclideanDistSq(a, b *[VectorDimsPad]uint8) int32 {
	var sum int32
	for i := 0; i < VectorDimsPad; i++ {
		d := int32(a[i]) - int32(b[i])
		sum += d * d
	}
	return sum
}
