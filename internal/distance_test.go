package internal

import (
	"math"
	"testing"
)

func quantizeArray(arr [VectorDimsPad]float32) [VectorDimsPad]uint8 {
	var res [VectorDimsPad]uint8
	for i, v := range arr {
		res[i] = QuantizeFloat32(v)
	}
	return res
}

func TestEuclideanDistSq_IdenticalVectors(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	aq := quantizeArray(a)
	
	var query [VectorDimsPad]float32
	for i, v := range aq {
		query[i] = DequantizeToFloat32(v)
	}

	got := EuclideanDistSq(&query, &aq)
	if got != 0 {
		t.Errorf("EuclideanDistSq(query, aq) = %v, want 0", got)
	}
}

func TestEuclideanDistSq_KnownDistance(t *testing.T) {
	a := [VectorDimsPad]float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b := [VectorDimsPad]float32{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	bq := quantizeArray(b)
	got := EuclideanDistSq(&a, &bq)
	if math.Abs(float64(got)-2.0) > 1e-3 {
		t.Errorf("EuclideanDistSq = %v, want 2.0", got)
	}
}

func TestEuclideanDistSq_AllDimensions(t *testing.T) {
	var a [VectorDimsPad]float32
	var b [VectorDimsPad]float32
	for i := 0; i < VectorDims; i++ {
		a[i] = 0.5
		b[i] = 0.6
	}
	bq := quantizeArray(b)
	got := EuclideanDistSq(&a, &bq)
	want := float32(VectorDims) * 0.01 // 14 * 0.01 = 0.14
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Errorf("EuclideanDistSq = %v, want %v", got, want)
	}
}

func TestEuclideanDistSq_PaddingIgnored(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	bq := quantizeArray(a)
	var query [VectorDimsPad]float32
	for i, v := range bq {
		query[i] = DequantizeToFloat32(v)
	}
	got := EuclideanDistSq(&query, &bq)
	if got != 0 {
		t.Errorf("EuclideanDistSq with zeroed padding = %v, want 0", got)
	}
}

func TestEuclideanDistSq_WithSentinelMinus1(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, -1, -1, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	aq := quantizeArray(a)
	var query [VectorDimsPad]float32
	for i, v := range aq {
		query[i] = DequantizeToFloat32(v)
	}
	got := EuclideanDistSq(&query, &aq)
	if got != 0 {
		t.Errorf("EuclideanDistSq with matching sentinels = %v, want 0", got)
	}

	c := [VectorDimsPad]float32{0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	d := [VectorDimsPad]float32{0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	dq := quantizeArray(d)
	got2 := EuclideanDistSq(&c, &dq)
	want := float32(2.25) // (-1 - 0.5)² = 2.25
	if math.Abs(float64(got2-want)) > 1e-4 {
		t.Errorf("EuclideanDistSq sentinel vs value = %v, want %v", got2, want)
	}
}

func TestEuclideanDistSq_Symmetry(t *testing.T) {
	a := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0.5, 0.1, 0, 0}
	b := [VectorDimsPad]float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 1, 1, 0.3, 0.9, 0, 0}
	aq := quantizeArray(a)
	bq := quantizeArray(b)
	ab := EuclideanDistSqRefRef(&aq, &bq)
	ba := EuclideanDistSqRefRef(&bq, &aq)
	if ab != ba {
		t.Errorf("EuclideanDistSqRefRef not symmetric: ab=%v, ba=%v", ab, ba)
	}
}
