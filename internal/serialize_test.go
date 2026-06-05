package internal

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSerializeDeserialize_SmallDataset(t *testing.T) {
	// Build a VP-Tree from the small dataset
	refs := smallReferenceDataset() // 10 vectors
	tree := BuildVPTree(refs)

	// Serialize to a temp file
	tmpFile := filepath.Join(t.TempDir(), "test_vptree.bin")
	if err := SerializeVPTree(tree, tmpFile); err != nil {
		t.Fatalf("SerializeVPTree failed: %v", err)
	}

	// Verify file was created
	fi, err := os.Stat(tmpFile)
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}
	t.Logf("serialized %d nodes to %d bytes", len(tree.Nodes), fi.Size())

	// Deserialize
	loaded, cleanup, err := LoadVPTreeBinary(tmpFile)
	if err != nil {
		t.Fatalf("LoadVPTreeBinary failed: %v", err)
	}
	defer cleanup()

	// Verify structure
	if len(loaded.Nodes) != len(tree.Nodes) {
		t.Errorf("nodes count: got %d, want %d", len(loaded.Nodes), len(tree.Nodes))
	}
	if len(loaded.Vectors) != len(tree.Vectors) {
		t.Errorf("vectors count: got %d, want %d", len(loaded.Vectors), len(tree.Vectors))
	}
	if len(loaded.Labels) != len(tree.Labels) {
		t.Errorf("labels count: got %d, want %d", len(loaded.Labels), len(tree.Labels))
	}
	if loaded.Root != tree.Root {
		t.Errorf("root: got %d, want %d", loaded.Root, tree.Root)
	}

	// Verify query results match
	queryFloats := [VectorDimsPad]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006, 0, 0}
	var query [VectorDimsPad]uint8
	for i, f := range queryFloats {
		query[i] = QuantizeFloat32(f)
	}
	origNeighbors := tree.KNN(&query, 5)
	loadedNeighbors := loaded.KNN(&query, 5)

	if origNeighbors.Len != loadedNeighbors.Len {
		t.Fatalf("neighbor count mismatch: orig=%d, loaded=%d", origNeighbors.Len, loadedNeighbors.Len)
	}

	for i := 0; i < origNeighbors.Len; i++ {
		origN := origNeighbors.Neighbors[i]
		loadedN := loadedNeighbors.Neighbors[i]
		if !approxEq32(origN.DistSq, loadedN.DistSq, 1e-4) {
			t.Errorf("neighbor[%d] dist: orig=%.6f, loaded=%.6f", i, origN.DistSq, loadedN.DistSq)
		}
		if origN.Label != loadedN.Label {
			t.Errorf("neighbor[%d] label: orig=%d, loaded=%d", i, origN.Label, loadedN.Label)
		}
	}
}

func TestSerializeDeserialize_ExampleDataset(t *testing.T) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("example references not found at %s", absPath)
	}

	// Load, build, serialize
	refs, err := LoadReferencesJSON(absPath)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	tree := BuildVPTree(refs)

	tmpFile := filepath.Join(t.TempDir(), "test_vptree.bin")
	if err := SerializeVPTree(tree, tmpFile); err != nil {
		t.Fatalf("serialize failed: %v", err)
	}

	fi, _ := os.Stat(tmpFile)
	t.Logf("serialized %d refs → %d bytes (%.1f KB)", len(refs), fi.Size(), float64(fi.Size())/1024)

	// Deserialize and validate
	loaded, cleanup, err := LoadVPTreeBinary(tmpFile)
	if err != nil {
		t.Fatalf("load binary failed: %v", err)
	}
	defer cleanup()

	// Test with spec examples
	// Legit tx
	vecLegit := Vectorize(&FraudRequest{
		ID: "tx-1329056812",
		Transaction: Transaction{Amount: 41.12, Installments: 2, RequestedAt: "2026-03-11T18:45:53Z"},
		Customer:    Customer{AvgAmount: 82.24, TxCount24h: 3, KnownMerchants: []string{"MERC-003", "MERC-016"}},
		Merchant:    Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal:    Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248},
	})
	var qLegit [VectorDimsPad]uint8
	for i, f := range vecLegit {
		qLegit[i] = QuantizeFloat32(f)
	}
	nLegit := loaded.KNN(&qLegit, 5)
	scoreLegit := ComputeFraudScore(nLegit)
	if !IsApproved(scoreLegit) {
		t.Errorf("legit tx should be approved via loaded tree, got score=%.1f", scoreLegit)
	}

	// Fraud tx
	vecFraud := Vectorize(&FraudRequest{
		ID: "tx-3330991687",
		Transaction: Transaction{Amount: 9505.97, Installments: 10, RequestedAt: "2026-03-14T05:15:12Z"},
		Customer:    Customer{AvgAmount: 81.28, TxCount24h: 20, KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"}},
		Merchant:    Merchant{ID: "MERC-068", MCC: "7802", AvgAmount: 54.86},
		Terminal:    Terminal{IsOnline: false, CardPresent: true, KmFromHome: 952.2745933273},
	})
	var qFraud [VectorDimsPad]uint8
	for i, f := range vecFraud {
		qFraud[i] = QuantizeFloat32(f)
	}
	nFraud := loaded.KNN(&qFraud, 5)
	scoreFraud := ComputeFraudScore(nFraud)
	if IsApproved(scoreFraud) {
		t.Errorf("fraud tx should be denied via loaded tree, got score=%.1f", scoreFraud)
	}

	t.Logf("legit: score=%.1f approved=%v | fraud: score=%.1f approved=%v",
		scoreLegit, IsApproved(scoreLegit), scoreFraud, IsApproved(scoreFraud))
}

func TestSerializeDeserialize_HeaderValidation(t *testing.T) {
	// Write a valid file first
	refs := smallReferenceDataset()
	tree := BuildVPTree(refs)
	tmpFile := filepath.Join(t.TempDir(), "test_vptree.bin")
	if err := SerializeVPTree(tree, tmpFile); err != nil {
		t.Fatalf("serialize failed: %v", err)
	}

	// Verify header fields by loading
	loaded, cleanup, err := LoadVPTreeBinary(tmpFile)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer cleanup()

	if loaded.Root != tree.Root {
		t.Errorf("root mismatch: got %d, want %d", loaded.Root, tree.Root)
	}
	if len(loaded.Nodes) != len(tree.Nodes) {
		t.Errorf("nodes mismatch: got %d, want %d", len(loaded.Nodes), len(tree.Nodes))
	}
	if len(loaded.Vectors) != len(tree.Vectors) {
		t.Errorf("vectors mismatch: got %d, want %d", len(loaded.Vectors), len(tree.Vectors))
	}
}

func TestDeserialize_InvalidFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "nonexistent.bin")
	_, _, err := LoadVPTreeBinary(tmpFile)
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestDeserialize_CorruptHeader(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "corrupt.bin")
	// Write garbage
	if err := os.WriteFile(tmpFile, []byte("not a vptree file"), 0644); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	_, _, err := LoadVPTreeBinary(tmpFile)
	if err == nil {
		t.Error("expected error for corrupt header")
	}
	t.Logf("got expected error: %v", err)
}

func TestDeserialize_TooSmall(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "small.bin")
	if err := os.WriteFile(tmpFile, []byte{0x01, 0x02}, 0644); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	_, _, err := LoadVPTreeBinary(tmpFile)
	if err == nil {
		t.Error("expected error for file too small")
	}
}
