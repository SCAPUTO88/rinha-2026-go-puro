package internal

// EuclideanDistSq calcula a distância euclidiana ao quadrado entre a query (float32) e uma referência (uint8).
func EuclideanDistSq(query *[VectorDimsPad]float32, ref *[VectorDimsPad]uint8) float32 {
	var sum float32
	for i := 0; i < VectorDimsPad; i++ {
		refVal := DequantizeToFloat32(ref[i])
		d := query[i] - refVal
		sum += d * d
	}
	return sum
}

// EuclideanDistSqRefRef calcula a distância euclidiana ao quadrado entre duas referências quantizadas (uint8).
func EuclideanDistSqRefRef(a, b *[VectorDimsPad]uint8) float32 {
	var sum float32
	for i := 0; i < VectorDimsPad; i++ {
		aVal := DequantizeToFloat32(a[i])
		bVal := DequantizeToFloat32(b[i])
		d := aVal - bVal
		sum += d * d
	}
	return sum
}
