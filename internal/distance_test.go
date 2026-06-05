package internal

import (
	"math"
	"testing"
)

func quantizeArray(arr [VectorDimsPad]float32) [VectorDimsPad]uint8 {
	var res [VectorDimsPad]uint8
	for i := 0; i < VectorDims; i++ {
		res[i] = QuantizeFloat32(arr[i])
	}
	return res
}

func TestEuclideanDistSq_IdenticalVectors(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	aq := quantizeArray(a)

	got := EuclideanDistSq(&aq, &aq)
	if got != 0 {
		t.Errorf("EuclideanDistSq(aq, aq) = %v, want 0", got)
	}
}

func TestEuclideanDistSq_KnownDistance(t *testing.T) {
	a := [VectorDimsPad]float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b := [VectorDimsPad]float32{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	aq := quantizeArray(a)
	bq := quantizeArray(b)
	got := EuclideanDistSq(&aq, &bq)
	realDistSq := float32(got) * 0.000064
	if math.Abs(float64(realDistSq)-2.0) > 2e-2 {
		t.Errorf("EuclideanDistSq = %v (real: %v), want 2.0", got, realDistSq)
	}
}

func TestEuclideanDistSq_AllDimensions(t *testing.T) {
	var a [VectorDimsPad]float32
	var b [VectorDimsPad]float32
	for i := 0; i < VectorDims; i++ {
		a[i] = 0.5
		b[i] = 0.6
	}
	aq := quantizeArray(a)
	bq := quantizeArray(b)
	got := EuclideanDistSq(&aq, &bq)
	realDistSq := float32(got) * 0.000064
	want := float32(VectorDims) * 0.01 // 14 * 0.01 = 0.14
	if math.Abs(float64(realDistSq-want)) > 2e-2 {
		t.Errorf("EuclideanDistSq = %v (real: %v), want %v", got, realDistSq, want)
	}
}

func TestEuclideanDistSq_PaddingIgnored(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	bq := quantizeArray(a)
	got := EuclideanDistSq(&bq, &bq)
	if got != 0 {
		t.Errorf("EuclideanDistSq with zeroed padding = %v, want 0", got)
	}
}

func TestEuclideanDistSq_WithSentinelMinus1(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, -1, -1, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	aq := quantizeArray(a)
	got := EuclideanDistSq(&aq, &aq)
	if got != 0 {
		t.Errorf("EuclideanDistSq with matching sentinels = %v, want 0", got)
	}

	c := [VectorDimsPad]float32{0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	d := [VectorDimsPad]float32{0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	cq := quantizeArray(c)
	dq := quantizeArray(d)
	got2 := EuclideanDistSq(&cq, &dq)
	realDistSq := float32(got2) * 0.000064
	want := float32(2.25) // (-1 - 0.5)² = 2.25
	if math.Abs(float64(realDistSq-want)) > 2e-2 {
		t.Errorf("EuclideanDistSq sentinel vs value = %v (real: %v), want %v", got2, realDistSq, want)
	}
}

func TestEuclideanDistSq_Symmetry(t *testing.T) {
	a := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0.5, 0.1, 0, 0}
	b := [VectorDimsPad]float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 1, 1, 0.3, 0.9, 0, 0}
	aq := quantizeArray(a)
	bq := quantizeArray(b)
	ab := EuclideanDistSq(&aq, &bq)
	ba := EuclideanDistSq(&bq, &aq)
	if ab != ba {
		t.Errorf("EuclideanDistSq not symmetric: ab=%v, ba=%v", ab, ba)
	}
}
