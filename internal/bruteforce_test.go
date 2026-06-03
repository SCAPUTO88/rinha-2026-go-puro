package internal

import (
	"math"
	"testing"
)

// Small dataset from example-references.json for brute-force tests
func smallReferenceDataset() []Reference {
	type rawRef struct {
		vec [VectorDimsPad]float32
		lbl uint8
	}
	raws := []rawRef{
		{vec: [VectorDimsPad]float32{0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0109, 0.1667, 0.05, 0.3913, 0.6667, 0.3007, 0.0139, 0.0154, 0.2, 0, 1, 0, 0.15, 0.0282, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0336, 0.1667, 0.05, 0.4348, 0.6667, 0.1278, 0.0008, 0.017, 0.1, 0, 1, 0, 0.2, 0.02, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0415, 0.25, 0.05, 0.7391, 1, 0.2375, 0.0121, 0.0005, 0.2, 0, 1, 0, 0.3, 0.0493, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.0291, 0.0833, 0.05, 0.3913, 0.3333, 0.3028, 0.0044, 0.028, 0.1, 0, 1, 0, 0.3, 0.043, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.5796, 0.9167, 1, 0.0435, 0, 0.0056, 0.4394, 0.4598, 0.4, 1, 0, 1, 0.85, 0.0032, 0, 0}, lbl: LabelFraud},
		{vec: [VectorDimsPad]float32{0.0035, 0.1667, 0.05, 0.4783, 0.8333, 0.2264, 0.001, 0.0488, 0.05, 0, 1, 0, 0.15, 0.0231, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.9708, 1, 1, 0.1304, 0.3333, -1, -1, 0.6657, 1, 1, 0, 1, 0.75, 0.0077, 0, 0}, lbl: LabelFraud},
		{vec: [VectorDimsPad]float32{0.0092, 0.0833, 0.05, 0.6522, 1, 0.0417, 0.0116, 0.0025, 0.1, 0, 1, 0, 0.15, 0.0101, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.3536, 0.5, 1, 0.087, 0.6667, 0.0049, 0.8445, 0.8925, 0.8, 1, 0, 1, 0.85, 0.0035, 0, 0}, lbl: LabelFraud},
	}
	refs := make([]Reference, len(raws))
	for i, r := range raws {
		var ref Reference
		for d := 0; d < VectorDimsPad; d++ {
			ref.Vector[d] = QuantizeFloat32(r.vec[d])
		}
		ref.Label = r.lbl
		refs[i] = ref
	}
	return refs
}

func TestBruteForceKNN_SmallDataset(t *testing.T) {
	refs := smallReferenceDataset()

	// Query: legit transaction vector from spec (tx-1329056812)
	query := [VectorDimsPad]float32{
		0.0041, 0.1667, 0.05, 0.7826, 0.3333,
		-1, -1,
		0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
		0, 0, // padding
	}

	neighbors := BruteForceKNN(&query, refs, 5)

	if len(neighbors) != 5 {
		t.Fatalf("BruteForceKNN returned %d neighbors, want 5", len(neighbors))
	}

	// All 5 nearest should be legit for this query (it's a clearly legit vector)
	fraudCount := 0
	for _, n := range neighbors {
		if n.Label == LabelFraud {
			fraudCount++
		}
	}

	var knnResult KNNResult
	copy(knnResult.Neighbors[:], neighbors)
	knnResult.Len = len(neighbors)
	score := ComputeFraudScore(knnResult)
	t.Logf("fraud_score = %.1f (%d fraud out of 5)", score, fraudCount)

	// With this small dataset and clearly legit query, expect mostly legit neighbors
	// The legit vectors are all close to this query; fraud vectors are far away
	if score >= 0.6 {
		t.Errorf("Expected legit query to have fraud_score < 0.6, got %.1f", score)
	}

	// Verify distances are sorted (ascending)
	for i := 1; i < len(neighbors); i++ {
		if neighbors[i].DistSq < neighbors[i-1].DistSq {
			t.Errorf("neighbors not sorted: [%d].dist=%v > [%d].dist=%v",
				i-1, neighbors[i-1].DistSq, i, neighbors[i].DistSq)
		}
	}
}

func TestBruteForceKNN_FraudQuery(t *testing.T) {
	refs := smallReferenceDataset()

	// Query: fraud transaction vector from spec (tx-3330991687)
	query := [VectorDimsPad]float32{
		0.9506, 0.8333, 1.0, 0.2174, 0.8333,
		-1, -1,
		0.9523, 1.0, 0, 1, 1, 0.75, 0.0055,
		0, 0, // padding
	}

	neighbors := BruteForceKNN(&query, refs, 5)

	if len(neighbors) != 5 {
		t.Fatalf("BruteForceKNN returned %d neighbors, want 5", len(neighbors))
	}

	// Log distances for debugging
	for i, n := range neighbors {
		label := "legit"
		if n.Label == LabelFraud {
			label = "fraud"
		}
		t.Logf("  neighbor[%d]: dist²=%.4f dist=%.4f %s", i, n.DistSq, math.Sqrt(float64(n.DistSq)), label)
	}

	// Verify distances are sorted
	for i := 1; i < len(neighbors); i++ {
		if neighbors[i].DistSq < neighbors[i-1].DistSq {
			t.Errorf("neighbors not sorted: [%d].dist=%v > [%d].dist=%v",
				i-1, neighbors[i-1].DistSq, i, neighbors[i].DistSq)
		}
	}
}

func TestBruteForceKNN_DatasetSmallerThanK(t *testing.T) {
	rawRefs := []struct {
		vec [VectorDimsPad]float32
		lbl uint8
	}{
		{vec: [VectorDimsPad]float32{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, lbl: LabelLegit},
		{vec: [VectorDimsPad]float32{0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, lbl: LabelFraud},
	}
	refs := make([]Reference, len(rawRefs))
	for i, r := range rawRefs {
		var ref Reference
		for d := 0; d < VectorDimsPad; d++ {
			ref.Vector[d] = QuantizeFloat32(r.vec[d])
		}
		ref.Label = r.lbl
		refs[i] = ref
	}
	query := [VectorDimsPad]float32{0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	neighbors := BruteForceKNN(&query, refs, 5)

	// Should return only 2 (dataset size), not 5
	if len(neighbors) != 2 {
		t.Fatalf("BruteForceKNN with 2 refs and k=5 returned %d, want 2", len(neighbors))
	}
}

func TestBruteForceKNN_ExactMatch(t *testing.T) {
	refs := smallReferenceDataset()

	// Query is exactly refs[0]
	var query [VectorDimsPad]float32
	for i, v := range refs[0].Vector {
		query[i] = DequantizeToFloat32(v)
	}

	neighbors := BruteForceKNN(&query, refs, 1)
	if len(neighbors) != 1 {
		t.Fatalf("got %d neighbors, want 1", len(neighbors))
	}
	if neighbors[0].DistSq != 0 {
		t.Errorf("exact match distance = %v, want 0", neighbors[0].DistSq)
	}
}
