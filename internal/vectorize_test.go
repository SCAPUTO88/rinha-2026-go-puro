package internal

import (
	"math"
	"testing"
)

// approxEqual compares two float32 values with a tolerance.
func approxEqual(a, b float32, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

// assertVectorApprox checks that two vectors match within tolerance.
func assertVectorApprox(t *testing.T, got, want [VectorDimsPad]float32, tol float32) {
	t.Helper()
	for i := 0; i < VectorDims; i++ {
		if !approxEqual(got[i], want[i], tol) {
			t.Errorf("dim[%d] = %.6f, want %.6f (diff=%.6f, tol=%.4f)",
				i, got[i], want[i], got[i]-want[i], tol)
		}
	}
	// Verify padding is zero
	for i := VectorDims; i < VectorDimsPad; i++ {
		if got[i] != 0 {
			t.Errorf("padding dim[%d] = %v, want 0", i, got[i])
		}
	}
}

// TestVectorize_LegitTransaction validates against the exact example from REGRAS_DE_DETECCAO.md
// Transaction tx-1329056812: legit, small amount, known merchant, near home, no last tx.
// Expected vector: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
func TestVectorize_LegitTransaction(t *testing.T) {
	req := &FraudRequest{
		ID: "tx-1329056812",
		Transaction: Transaction{
			Amount:      41.12,
			Installments: 2,
			RequestedAt: "2026-03-11T18:45:53Z",
		},
		Customer: Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{
			ID:        "MERC-016",
			MCC:       "5411",
			AvgAmount: 60.25,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  29.2331036248,
		},
		LastTransaction: nil, // null
	}

	got := Vectorize(req)

	want := [VectorDimsPad]float32{
		0.0041,  // [0]  amount: 41.12 / 10000
		0.1667,  // [1]  installments: 2 / 12
		0.05,    // [2]  amount_vs_avg: (41.12/82.24)/10 = 0.05
		0.7826,  // [3]  hour: 18/23
		0.3333,  // [4]  day_of_week: wed=2, 2/6
		-1,      // [5]  minutes_since_last_tx: null → -1
		-1,      // [6]  km_from_last_tx: null → -1
		0.0292,  // [7]  km_from_home: 29.23/1000
		0.15,    // [8]  tx_count_24h: 3/20
		0,       // [9]  is_online: false → 0
		1,       // [10] card_present: true → 1
		0,       // [11] unknown_merchant: MERC-016 in known → 0
		0.15,    // [12] mcc_risk: 5411 → 0.15
		0.006,   // [13] merchant_avg_amount: 60.25/10000
		0, 0,    // [14-15] padding
	}

	assertVectorApprox(t, got, want, 0.001)
}

// TestVectorize_FraudTransaction validates against the fraud example from REGRAS_DE_DETECCAO.md
// Transaction tx-3330991687: fraud, high amount, unknown merchant, far from home, no last tx.
// Expected vector: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
func TestVectorize_FraudTransaction(t *testing.T) {
	req := &FraudRequest{
		ID: "tx-3330991687",
		Transaction: Transaction{
			Amount:      9505.97,
			Installments: 10,
			RequestedAt: "2026-03-14T05:15:12Z",
		},
		Customer: Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: Merchant{
			ID:        "MERC-068",
			MCC:       "7802",
			AvgAmount: 54.86,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  952.2745933273,
		},
		LastTransaction: nil,
	}

	got := Vectorize(req)

	want := [VectorDimsPad]float32{
		0.9506,  // [0]  amount: 9505.97 / 10000
		0.8333,  // [1]  installments: 10 / 12
		1.0,     // [2]  amount_vs_avg: (9505.97/81.28)/10 = 11.69 → clamped 1.0
		0.2174,  // [3]  hour: 5/23
		0.8333,  // [4]  day_of_week: sat=5, 5/6
		-1,      // [5]  null → -1
		-1,      // [6]  null → -1
		0.9523,  // [7]  km_from_home: 952.27/1000
		1.0,     // [8]  tx_count_24h: 20/20
		0,       // [9]  is_online: false → 0
		1,       // [10] card_present: true → 1
		1,       // [11] unknown_merchant: MERC-068 not in known → 1
		0.75,    // [12] mcc_risk: 7802 → 0.75
		0.0055,  // [13] merchant_avg_amount: 54.86/10000
		0, 0,    // [14-15] padding
	}

	assertVectorApprox(t, got, want, 0.001)
}

// TestVectorize_WithLastTransaction validates that dims 5 and 6 are calculated
// when last_transaction is present (not null).
func TestVectorize_WithLastTransaction(t *testing.T) {
	req := &FraudRequest{
		ID: "tx-3576980410",
		Transaction: Transaction{
			Amount:      384.88,
			Installments: 3,
			RequestedAt: "2026-03-11T20:23:35Z",
		},
		Customer: Customer{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-001", "MERC-001"},
		},
		Merchant: Merchant{
			ID:        "MERC-001",
			MCC:       "5912",
			AvgAmount: 298.95,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  13.7090520965,
		},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.8626479774,
		},
	}

	got := Vectorize(req)

	// Verify dims 5 and 6 are NOT -1
	if got[5] == -1 {
		t.Error("dim[5] should not be -1 when last_transaction is present")
	}
	if got[6] == -1 {
		t.Error("dim[6] should not be -1 when last_transaction is present")
	}

	// dim[5] = minutes_since_last_tx: minutes between 14:58:35 and 20:23:35 = 325 min
	// 325 / 1440 = 0.2257
	if !approxEqual(got[5], 0.2257, 0.001) {
		t.Errorf("dim[5] minutes_since_last = %.6f, want ~0.2257", got[5])
	}

	// dim[6] = km_from_last: 18.8626 / 1000 = 0.0189
	if !approxEqual(got[6], 0.0189, 0.001) {
		t.Errorf("dim[6] km_from_last = %.6f, want ~0.0189", got[6])
	}

	// dim[0] = amount: 384.88 / 10000 = 0.0385
	if !approxEqual(got[0], 0.0385, 0.001) {
		t.Errorf("dim[0] amount = %.6f, want ~0.0385", got[0])
	}

	// dim[11] = unknown_merchant: MERC-001 IS in known_merchants → 0
	if got[11] != 0 {
		t.Errorf("dim[11] unknown_merchant = %v, want 0 (MERC-001 is known)", got[11])
	}

	// dim[12] = mcc_risk: 5912 → 0.20
	if !approxEqual(got[12], 0.20, 0.001) {
		t.Errorf("dim[12] mcc_risk = %v, want 0.20", got[12])
	}
}

// --- Clamp Tests ---

func TestClamp_BelowZero(t *testing.T) {
	if got := clamp(-0.5); got != 0.0 {
		t.Errorf("clamp(-0.5) = %v, want 0.0", got)
	}
}

func TestClamp_AboveOne(t *testing.T) {
	if got := clamp(1.25); got != 1.0 {
		t.Errorf("clamp(1.25) = %v, want 1.0", got)
	}
}

func TestClamp_InRange(t *testing.T) {
	if got := clamp(0.5); got != 0.5 {
		t.Errorf("clamp(0.5) = %v, want 0.5", got)
	}
}

func TestClamp_ExactBounds(t *testing.T) {
	if got := clamp(0.0); got != 0.0 {
		t.Errorf("clamp(0.0) = %v, want 0.0", got)
	}
	if got := clamp(1.0); got != 1.0 {
		t.Errorf("clamp(1.0) = %v, want 1.0", got)
	}
}

// --- Weekday Tests ---
// The spec uses: Monday=0, Sunday=6

func TestWeekday_Wednesday(t *testing.T) {
	// 2026-03-11 is a Wednesday → 2
	got := isoWeekday("2026-03-11T18:45:53Z")
	if got != 2 {
		t.Errorf("weekday for 2026-03-11 = %v, want 2 (Wednesday)", got)
	}
}

func TestWeekday_Saturday(t *testing.T) {
	// 2026-03-14 is a Saturday → 5
	got := isoWeekday("2026-03-14T05:15:12Z")
	if got != 5 {
		t.Errorf("weekday for 2026-03-14 = %v, want 5 (Saturday)", got)
	}
}

func TestWeekday_Sunday(t *testing.T) {
	// 2026-03-15 is a Sunday → 6
	got := isoWeekday("2026-03-15T08:40:05Z")
	if got != 6 {
		t.Errorf("weekday for 2026-03-15 = %v, want 6 (Sunday)", got)
	}
}

func TestWeekday_Monday(t *testing.T) {
	// 2026-03-16 is a Monday → 0
	got := isoWeekday("2026-03-16T10:00:00Z")
	if got != 0 {
		t.Errorf("weekday for 2026-03-16 = %v, want 0 (Monday)", got)
	}
}
