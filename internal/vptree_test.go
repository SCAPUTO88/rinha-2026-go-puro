package internal

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestVPTree_Build_SmallDataset(t *testing.T) {
	refs := smallReferenceDataset() // 10 vectors from Phase 3 tests
	tree := BuildVPTree(refs)

	if tree == nil {
		t.Fatal("BuildVPTree returned nil")
	}
	if len(tree.Nodes) == 0 {
		t.Fatal("BuildVPTree produced 0 nodes")
	}
	// A balanced binary tree of 10 elements has about 10 nodes
	t.Logf("built VP-Tree with %d nodes from %d references", len(tree.Nodes), len(refs))
}

func TestVPTree_KNN_MatchesBruteForce_SmallDataset(t *testing.T) {
	refs := smallReferenceDataset() // 10 vectors
	tree := BuildVPTree(refs)

	// Test with multiple query vectors
	queries := [][VectorDimsPad]float32{
		// Legit query (from spec)
		{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006, 0, 0},
		// Fraud query (from spec)
		{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055, 0, 0},
		// Middle-ground query
		{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0},
		// Near-zero query
		{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0.1, 0.01, 0, 0},
	}

	for qi, query := range queries {
		bfNeighbors := BruteForceKNN(&query, refs, 5)
		vpNeighbors := tree.KNN(&query, 5)

		if vpNeighbors.Len != len(bfNeighbors) {
			t.Errorf("query[%d]: VP-Tree returned %d neighbors, brute force returned %d",
				qi, vpNeighbors.Len, len(bfNeighbors))
			continue
		}

		// Verify distances match (both use squared euclidean)
		for i := 0; i < vpNeighbors.Len; i++ {
			vpN := vpNeighbors.Neighbors[i]
			bfN := bfNeighbors[i]
			if !approxEq32(vpN.DistSq, bfN.DistSq, 1e-4) {
				t.Errorf("query[%d] neighbor[%d]: VP dist²=%.6f, BF dist²=%.6f",
					qi, i, vpN.DistSq, bfN.DistSq)
			}
			if vpN.Label != bfN.Label {
				t.Errorf("query[%d] neighbor[%d]: VP label=%d, BF label=%d",
					qi, i, vpN.Label, bfN.Label)
			}
		}

		// Verify fraud scores match
		var bfRes KNNResult
		copy(bfRes.Neighbors[:], bfNeighbors)
		bfRes.Len = len(bfNeighbors)

		bfScore := ComputeFraudScore(bfRes)
		vpScore := ComputeFraudScore(vpNeighbors)
		if bfScore != vpScore {
			t.Errorf("query[%d]: VP score=%.1f, BF score=%.1f", qi, vpScore, bfScore)
		}
	}
}

func TestVPTree_KNN_MatchesBruteForce_ExampleDataset(t *testing.T) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("example references file not found at %s", absPath)
	}

	refs, err := LoadReferencesJSON(absPath)
	if err != nil {
		t.Fatalf("LoadReferencesJSON failed: %v", err)
	}
	t.Logf("loaded %d references", len(refs))

	tree := BuildVPTree(refs)
	t.Logf("built VP-Tree with %d nodes", len(tree.Nodes))

	// Generate test queries: use every 10th reference vector as a query
	mismatchCount := 0
	totalQueries := 0
	for i := 0; i < len(refs); i += 10 {
		// Desquantiza para criar a query float32
		var query [VectorDimsPad]float32
		for d, v := range refs[i].Vector {
			query[d] = DequantizeToFloat32(v)
		}
		totalQueries++

		bfNeighbors := BruteForceKNN(&query, refs, 5)
		vpNeighbors := tree.KNN(&query, 5)

		if vpNeighbors.Len != len(bfNeighbors) {
			t.Errorf("query[%d]: VP returned %d, BF returned %d", i, vpNeighbors.Len, len(bfNeighbors))
			mismatchCount++
			continue
		}

		// Compare fraud scores (the final output that matters)
		var bfRes KNNResult
		copy(bfRes.Neighbors[:], bfNeighbors)
		bfRes.Len = len(bfNeighbors)

		bfScore := ComputeFraudScore(bfRes)
		vpScore := ComputeFraudScore(vpNeighbors)
		if bfScore != vpScore {
			t.Errorf("query[%d]: VP score=%.1f, BF score=%.1f", i, vpScore, bfScore)
			mismatchCount++
		}

		// Also verify distances match
		for j := 0; j < vpNeighbors.Len; j++ {
			vpN := vpNeighbors.Neighbors[j]
			bfN := bfNeighbors[j]
			if !approxEq32(vpN.DistSq, bfN.DistSq, 1e-3) {
				t.Errorf("query[%d] neighbor[%d]: VP dist²=%.6f, BF dist²=%.6f",
					i, j, vpN.DistSq, bfN.DistSq)
				mismatchCount++
				break
			}
		}
	}

	t.Logf("tested %d queries, %d mismatches", totalQueries, mismatchCount)
	if mismatchCount > 0 {
		t.Errorf("VP-Tree produced %d mismatches out of %d queries", mismatchCount, totalQueries)
	}
}

func TestVPTree_KNN_SpecExample_Legit(t *testing.T) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("example references file not found at %s", absPath)
	}

	refs, err := LoadReferencesJSON(absPath)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	tree := BuildVPTree(refs)

	// Legit transaction from spec (tx-1329056812)
	req := &FraudRequest{
		ID: "tx-1329056812",
		Transaction: Transaction{
			Amount: 41.12, Installments: 2, RequestedAt: "2026-03-11T18:45:53Z",
		},
		Customer: Customer{
			AvgAmount: 82.24, TxCount24h: 3, KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal: Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248},
		LastTransaction: nil,
	}

	vec := Vectorize(req)
	neighbors := tree.KNN(&vec, 5)
	score := ComputeFraudScore(neighbors)
	approved := IsApproved(score)

	t.Logf("Legit tx: fraud_score=%.1f, approved=%v", score, approved)
	for i := 0; i < neighbors.Len; i++ {
		n := neighbors.Neighbors[i]
		label := "legit"
		if n.Label == LabelFraud {
			label = "fraud"
		}
		t.Logf("  [%d] dist=%.4f %s", i, math.Sqrt(float64(n.DistSq)), label)
	}

	if !approved {
		t.Errorf("legit transaction should be approved, got score=%.1f", score)
	}
}

func TestVPTree_KNN_SpecExample_Fraud(t *testing.T) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("example references file not found at %s", absPath)
	}

	refs, err := LoadReferencesJSON(absPath)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	tree := BuildVPTree(refs)

	// Fraud transaction from spec (tx-3330991687)
	req := &FraudRequest{
		ID: "tx-3330991687",
		Transaction: Transaction{
			Amount: 9505.97, Installments: 10, RequestedAt: "2026-03-14T05:15:12Z",
		},
		Customer: Customer{
			AvgAmount: 81.28, TxCount24h: 20, KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: Merchant{ID: "MERC-068", MCC: "7802", AvgAmount: 54.86},
		Terminal: Terminal{IsOnline: false, CardPresent: true, KmFromHome: 952.2745933273},
		LastTransaction: nil,
	}

	vec := Vectorize(req)
	neighbors := tree.KNN(&vec, 5)
	score := ComputeFraudScore(neighbors)
	approved := IsApproved(score)

	t.Logf("Fraud tx: fraud_score=%.1f, approved=%v", score, approved)
	for i := 0; i < neighbors.Len; i++ {
		n := neighbors.Neighbors[i]
		label := "legit"
		if n.Label == LabelFraud {
			label = "fraud"
		}
		t.Logf("  [%d] dist=%.4f %s", i, math.Sqrt(float64(n.DistSq)), label)
	}

	if approved {
		t.Errorf("fraud transaction should be denied, got score=%.1f", score)
	}
}

func TestVPTree_KNN_SingleElement(t *testing.T) {
	var v [VectorDimsPad]uint8
	rawFloats := []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.3, 0.05, 0, 0}
	for i, f := range rawFloats {
		v[i] = QuantizeFloat32(f)
	}
	refs := []Reference{
		{Vector: v, Label: LabelLegit},
	}
	tree := BuildVPTree(refs)

	query := [VectorDimsPad]float32{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 1, 0, 0.3, 0.05, 0, 0}
	neighbors := tree.KNN(&query, 5)

	if neighbors.Len != 1 {
		t.Fatalf("expected 1 neighbor from single-element tree, got %d", neighbors.Len)
	}
	if neighbors.Neighbors[0].Label != LabelLegit {
		t.Errorf("expected legit label, got %d", neighbors.Neighbors[0].Label)
	}
}

func TestVPTree_KNN_EmptyDataset(t *testing.T) {
	refs := []Reference{}
	tree := BuildVPTree(refs)

	query := [VectorDimsPad]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.3, 0.05, 0, 0}
	neighbors := tree.KNN(&query, 5)

	if neighbors.Len != 0 {
		t.Errorf("expected 0 neighbors from empty tree, got %d", neighbors.Len)
	}
}
