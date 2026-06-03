package internal

import (
	"math"
	"testing"
)

func TestEuclideanDistSq_IdenticalVectors(t *testing.T) {
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	got := EuclideanDistSq(&a, &a)
	if got != 0 {
		t.Errorf("EuclideanDistSq(a, a) = %v, want 0", got)
	}
}

func TestEuclideanDistSq_KnownDistance(t *testing.T) {
	// Simple case: a = [1, 0, 0, ...], b = [0, 1, 0, ...]
	// dist² = (1-0)² + (0-1)² = 2.0
	a := [VectorDimsPad]float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b := [VectorDimsPad]float32{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got := EuclideanDistSq(&a, &b)
	if math.Abs(float64(got)-2.0) > 1e-6 {
		t.Errorf("EuclideanDistSq = %v, want 2.0", got)
	}
}

func TestEuclideanDistSq_AllDimensions(t *testing.T) {
	// All 14 dims differ by 0.1 each: dist² = 14 * 0.01 = 0.14
	var a, b [VectorDimsPad]float32
	for i := 0; i < VectorDims; i++ {
		a[i] = 0.5
		b[i] = 0.6
	}
	got := EuclideanDistSq(&a, &b)
	want := float32(VectorDims) * 0.01 // 14 * 0.01 = 0.14
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("EuclideanDistSq = %v, want %v", got, want)
	}
}

func TestEuclideanDistSq_PaddingIgnored(t *testing.T) {
	// Padding dims are always zero, so they don't affect the distance
	a := [VectorDimsPad]float32{0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b := [VectorDimsPad]float32{0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got := EuclideanDistSq(&a, &b)
	if got != 0 {
		t.Errorf("EuclideanDistSq with zeroed padding = %v, want 0", got)
	}
}

func TestEuclideanDistSq_WithSentinelMinus1(t *testing.T) {
	// Vectors with -1 sentinel in dims 5 and 6 (last_transaction = null)
	// Both have -1 → distance on those dims is 0
	a := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, -1, -1, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	b := [VectorDimsPad]float32{0.5, 0.3, 0.1, 0.8, 0.2, -1, -1, 0.7, 0.9, 1, 0, 1, 0.5, 0.3, 0, 0}
	got := EuclideanDistSq(&a, &b)
	if got != 0 {
		t.Errorf("EuclideanDistSq with matching sentinels = %v, want 0", got)
	}

	// One has -1, other has 0.5 → dist on dim5 = (-1 - 0.5)² = 2.25
	c := [VectorDimsPad]float32{0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	d := [VectorDimsPad]float32{0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got2 := EuclideanDistSq(&c, &d)
	want := float32(2.25) // (-1 - 0.5)² = 2.25
	if math.Abs(float64(got2-want)) > 1e-5 {
		t.Errorf("EuclideanDistSq sentinel vs value = %v, want %v", got2, want)
	}
}

func TestEuclideanDistSq_Symmetry(t *testing.T) {
	a := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0.5, 0.1, 0, 0}
	b := [VectorDimsPad]float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 1, 1, 0.3, 0.9, 0, 0}
	ab := EuclideanDistSq(&a, &b)
	ba := EuclideanDistSq(&b, &a)
	if ab != ba {
		t.Errorf("EuclideanDistSq not symmetric: ab=%v, ba=%v", ab, ba)
	}
}
