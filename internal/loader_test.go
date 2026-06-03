package internal

import (
	"os"
	"path/filepath"
	"testing"
)

// Path to the example references file in the challenge repo (read-only).
const exampleRefsPath = "../rinha-de-backend-2026/resources/example-references.json"
const fullRefsGzPath = "../rinha-de-backend-2026/resources/references.json.gz"

func TestLoadReferences_ExampleFile(t *testing.T) {
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

	// The example file should have some entries
	if len(refs) == 0 {
		t.Fatal("loaded 0 references from example file")
	}
	t.Logf("loaded %d references from example file", len(refs))

	// Verify vector dimensions: all 14 real dims should have values, padding should be 0
	for i, ref := range refs {
		if ref.Vector[14] != 0 || ref.Vector[15] != 0 {
			t.Errorf("ref[%d] padding not zero: [14]=%v [15]=%v", i, ref.Vector[14], ref.Vector[15])
			break
		}
		// Label should be valid
		if ref.Label != LabelLegit && ref.Label != LabelFraud {
			t.Errorf("ref[%d] invalid label: %v", i, ref.Label)
			break
		}
	}

	// Count fraud vs legit
	fraudCount := 0
	for _, ref := range refs {
		if ref.Label == LabelFraud {
			fraudCount++
		}
	}
	legitCount := len(refs) - fraudCount
	t.Logf("fraud: %d (%.1f%%), legit: %d (%.1f%%)",
		fraudCount, float64(fraudCount)/float64(len(refs))*100,
		legitCount, float64(legitCount)/float64(len(refs))*100)

	// Verify first reference matches what we saw in the file
	// First entry: legit, vector[0] = 0.01
	if refs[0].Label != LabelLegit {
		t.Errorf("first ref should be legit, got label=%v", refs[0].Label)
	}
	if !approxEq32(refs[0].Vector[0], 0.01, 0.001) {
		t.Errorf("first ref vector[0] = %v, want ~0.01", refs[0].Vector[0])
	}
}

func TestLoadReferencesGzip_FullFile(t *testing.T) {
	absPath, err := filepath.Abs(fullRefsGzPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("full references.json.gz not found at %s (skip in CI)", absPath)
	}

	// This test loads 3M vectors — it's slow, skip unless explicitly requested
	if testing.Short() {
		t.Skip("skipping full dataset load in short mode")
	}

	refs, err := LoadReferencesGzip(absPath)
	if err != nil {
		t.Fatalf("LoadReferencesGzip failed: %v", err)
	}

	// Should be exactly 3,000,000 references
	if len(refs) != 3_000_000 {
		t.Errorf("loaded %d references, want 3,000,000", len(refs))
	}
	t.Logf("loaded %d references from gzip file", len(refs))

	// Verify ~35% are fraud (from generate-data.sh: --fraud-ratio-refs 0.35)
	fraudCount := 0
	for _, ref := range refs {
		if ref.Label == LabelFraud {
			fraudCount++
		}
	}
	fraudRate := float64(fraudCount) / float64(len(refs))
	t.Logf("fraud rate: %.2f%% (%d/%d)", fraudRate*100, fraudCount, len(refs))
	if fraudRate < 0.30 || fraudRate > 0.40 {
		t.Errorf("fraud rate %.2f%% outside expected range [30%%, 40%%]", fraudRate*100)
	}
}

func TestEndToEnd_BruteForce_ExampleDataset(t *testing.T) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		t.Fatalf("failed to resolve path: %v", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		t.Skipf("example references file not found at %s", absPath)
	}

	// Load references
	refs, err := LoadReferencesJSON(absPath)
	if err != nil {
		t.Fatalf("LoadReferencesJSON failed: %v", err)
	}
	t.Logf("loaded %d references", len(refs))

	// Vectorize the legit transaction from spec (tx-1329056812)
	reqLegit := &FraudRequest{
		ID: "tx-1329056812",
		Transaction: Transaction{
			Amount: 41.12, Installments: 2,
			RequestedAt: "2026-03-11T18:45:53Z",
		},
		Customer: Customer{
			AvgAmount: 82.24, TxCount24h: 3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{
			ID: "MERC-016", MCC: "5411", AvgAmount: 60.25,
		},
		Terminal: Terminal{
			IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248,
		},
		LastTransaction: nil,
	}

	vecLegit := Vectorize(reqLegit)
	neighbors := BruteForceKNN(&vecLegit, refs, 5)
	score := ComputeFraudScore(neighbors)
	approved := IsApproved(score)

	t.Logf("Legit tx: fraud_score=%.1f, approved=%v", score, approved)
	for i, n := range neighbors {
		label := "legit"
		if n.Label == LabelFraud {
			label = "fraud"
		}
		t.Logf("  neighbor[%d]: dist²=%.6f %s", i, n.DistSq, label)
	}

	// This is a clearly legit transaction — expect approved
	if !approved {
		t.Errorf("legit transaction should be approved, got fraud_score=%.1f", score)
	}

	// Vectorize the fraud transaction from spec (tx-3330991687)
	reqFraud := &FraudRequest{
		ID: "tx-3330991687",
		Transaction: Transaction{
			Amount: 9505.97, Installments: 10,
			RequestedAt: "2026-03-14T05:15:12Z",
		},
		Customer: Customer{
			AvgAmount: 81.28, TxCount24h: 20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: Merchant{
			ID: "MERC-068", MCC: "7802", AvgAmount: 54.86,
		},
		Terminal: Terminal{
			IsOnline: false, CardPresent: true, KmFromHome: 952.2745933273,
		},
		LastTransaction: nil,
	}

	vecFraud := Vectorize(reqFraud)
	neighborsFraud := BruteForceKNN(&vecFraud, refs, 5)
	scoreFraud := ComputeFraudScore(neighborsFraud)
	approvedFraud := IsApproved(scoreFraud)

	t.Logf("Fraud tx: fraud_score=%.1f, approved=%v", scoreFraud, approvedFraud)
	for i, n := range neighborsFraud {
		label := "legit"
		if n.Label == LabelFraud {
			label = "fraud"
		}
		t.Logf("  neighbor[%d]: dist²=%.6f %s", i, n.DistSq, label)
	}

	// This is a clearly fraudulent transaction — expect denied
	if approvedFraud {
		t.Errorf("fraud transaction should be denied, got fraud_score=%.1f", scoreFraud)
	}
}

// approxEq32 compares two float32 values with tolerance.
func approxEq32(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}
