package internal

import (
	"os"
	"path/filepath"
	"strings"
	"net/http/httptest"
	"testing"
)

// BenchmarkEuclideanDistSq measures the scalar distance function.
func BenchmarkEuclideanDistSq(b *testing.B) {
	a := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1, 0, 0.3, 0.05, 0, 0}
	c := [VectorDimsPad]float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1, 0, 1, 0.7, 0.95, 0, 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EuclideanDistSq(&a, &c)
	}
}

// BenchmarkVectorize measures transaction vectorization.
func BenchmarkVectorize(b *testing.B) {
	req := &FraudRequest{
		ID: "tx-bench",
		Transaction: Transaction{Amount: 41.12, Installments: 2, RequestedAt: "2026-03-11T18:45:53Z"},
		Customer:    Customer{AvgAmount: 82.24, TxCount24h: 3, KnownMerchants: []string{"MERC-003", "MERC-016"}},
		Merchant:    Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal:    Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Vectorize(req)
	}
}

// BenchmarkVPTreeKNN_100 measures KNN on 100-vector VP-Tree.
func BenchmarkVPTreeKNN_100(b *testing.B) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		b.Skip("example references not found")
	}

	refs, _ := LoadReferencesJSON(absPath)
	tree := BuildVPTree(refs)
	query := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1, 0, 0.3, 0.05, 0, 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tree.KNN(&query, 5)
	}
}

// BenchmarkVPTreeKNN_10 measures KNN on 10-vector VP-Tree.
func BenchmarkVPTreeKNN_10(b *testing.B) {
	refs := smallReferenceDataset()
	tree := BuildVPTree(refs)
	query := [VectorDimsPad]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1, 0, 0.3, 0.05, 0, 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tree.KNN(&query, 5)
	}
}

// BenchmarkFullPipeline measures the complete hot path:
// JSON decode → vectorize → KNN → score → response
func BenchmarkFullPipeline(b *testing.B) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		b.Skip("example references not found")
	}

	refs, _ := LoadReferencesJSON(absPath)
	tree := BuildVPTree(refs)
	handler := NewFraudHandler(tree)
	mux := handler.RegisterRoutes()

	payload := `{"id":"tx-bench","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.2331036248},"last_transaction":null}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader(payload))
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
	}
}

// BenchmarkFullPipeline_Allocs is the same but reports allocations.
func BenchmarkFullPipeline_Allocs(b *testing.B) {
	absPath, err := filepath.Abs(exampleRefsPath)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		b.Skip("example references not found")
	}

	refs, _ := LoadReferencesJSON(absPath)
	tree := BuildVPTree(refs)
	handler := NewFraudHandler(tree)
	mux := handler.RegisterRoutes()

	payload := `{"id":"tx-bench","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.2331036248},"last_transaction":null}`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader(payload))
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
	}
}
