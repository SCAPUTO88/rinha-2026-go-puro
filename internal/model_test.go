package internal

import (
	"encoding/json"
	"math"
	"testing"
)

// --- FraudRequest Deserialization Tests ---

func TestFraudRequest_UnmarshalJSON_WithLastTransaction(t *testing.T) {
	payload := `{
		"id": "tx-3576980410",
		"transaction": {
			"amount": 384.88,
			"installments": 3,
			"requested_at": "2026-03-11T20:23:35Z"
		},
		"customer": {
			"avg_amount": 769.76,
			"tx_count_24h": 3,
			"known_merchants": ["MERC-009", "MERC-001", "MERC-001"]
		},
		"merchant": {
			"id": "MERC-001",
			"mcc": "5912",
			"avg_amount": 298.95
		},
		"terminal": {
			"is_online": false,
			"card_present": true,
			"km_from_home": 13.7090520965
		},
		"last_transaction": {
			"timestamp": "2026-03-11T14:58:35Z",
			"km_from_current": 18.8626479774
		}
	}`

	var req FraudRequest
	if err := json.Unmarshal([]byte(payload), &req); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	// ID
	if req.ID != "tx-3576980410" {
		t.Errorf("ID = %q, want %q", req.ID, "tx-3576980410")
	}

	// Transaction
	if req.Transaction.Amount != 384.88 {
		t.Errorf("Amount = %v, want %v", req.Transaction.Amount, 384.88)
	}
	if req.Transaction.Installments != 3 {
		t.Errorf("Installments = %v, want %v", req.Transaction.Installments, 3)
	}
	if req.Transaction.RequestedAt != "2026-03-11T20:23:35Z" {
		t.Errorf("RequestedAt = %q, want %q", req.Transaction.RequestedAt, "2026-03-11T20:23:35Z")
	}

	// Customer
	if req.Customer.AvgAmount != 769.76 {
		t.Errorf("AvgAmount = %v, want %v", req.Customer.AvgAmount, 769.76)
	}
	if req.Customer.TxCount24h != 3 {
		t.Errorf("TxCount24h = %v, want %v", req.Customer.TxCount24h, 3)
	}
	if len(req.Customer.KnownMerchants) != 3 {
		t.Errorf("KnownMerchants len = %v, want %v", len(req.Customer.KnownMerchants), 3)
	}

	// Merchant
	if req.Merchant.ID != "MERC-001" {
		t.Errorf("Merchant.ID = %q, want %q", req.Merchant.ID, "MERC-001")
	}
	if req.Merchant.MCC != "5912" {
		t.Errorf("Merchant.MCC = %q, want %q", req.Merchant.MCC, "5912")
	}
	if req.Merchant.AvgAmount != 298.95 {
		t.Errorf("Merchant.AvgAmount = %v, want %v", req.Merchant.AvgAmount, 298.95)
	}

	// Terminal
	if req.Terminal.IsOnline != false {
		t.Errorf("IsOnline = %v, want false", req.Terminal.IsOnline)
	}
	if req.Terminal.CardPresent != true {
		t.Errorf("CardPresent = %v, want true", req.Terminal.CardPresent)
	}
	if math.Abs(req.Terminal.KmFromHome-13.7090520965) > 1e-6 {
		t.Errorf("KmFromHome = %v, want %v", req.Terminal.KmFromHome, 13.7090520965)
	}

	// LastTransaction (non-null)
	if req.LastTransaction == nil {
		t.Fatal("LastTransaction should not be nil")
	}
	if req.LastTransaction.Timestamp != "2026-03-11T14:58:35Z" {
		t.Errorf("LastTransaction.Timestamp = %q, want %q", req.LastTransaction.Timestamp, "2026-03-11T14:58:35Z")
	}
	if math.Abs(req.LastTransaction.KmFromCurrent-18.8626479774) > 1e-6 {
		t.Errorf("KmFromCurrent = %v, want %v", req.LastTransaction.KmFromCurrent, 18.8626479774)
	}
}

func TestFraudRequest_UnmarshalJSON_WithNullLastTransaction(t *testing.T) {
	payload := `{
		"id": "tx-1329056812",
		"transaction": {
			"amount": 41.12,
			"installments": 2,
			"requested_at": "2026-03-11T18:45:53Z"
		},
		"customer": {
			"avg_amount": 82.24,
			"tx_count_24h": 3,
			"known_merchants": ["MERC-003", "MERC-016"]
		},
		"merchant": {
			"id": "MERC-016",
			"mcc": "5411",
			"avg_amount": 60.25
		},
		"terminal": {
			"is_online": false,
			"card_present": true,
			"km_from_home": 29.2331036248
		},
		"last_transaction": null
	}`

	var req FraudRequest
	if err := json.Unmarshal([]byte(payload), &req); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if req.ID != "tx-1329056812" {
		t.Errorf("ID = %q, want %q", req.ID, "tx-1329056812")
	}
	if req.LastTransaction != nil {
		t.Errorf("LastTransaction should be nil when JSON is null, got %+v", req.LastTransaction)
	}
	if len(req.Customer.KnownMerchants) != 2 {
		t.Errorf("KnownMerchants len = %v, want 2", len(req.Customer.KnownMerchants))
	}
}

// --- FraudResponse Serialization Tests ---

func TestFraudResponse_MarshalJSON_Denied(t *testing.T) {
	resp := FraudResponse{Approved: false, FraudScore: 1.0}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	// Unmarshal back to verify roundtrip
	var got FraudResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if got.Approved != false {
		t.Errorf("Approved = %v, want false", got.Approved)
	}
	if got.FraudScore != 1.0 {
		t.Errorf("FraudScore = %v, want 1.0", got.FraudScore)
	}
}

func TestFraudResponse_MarshalJSON_Approved(t *testing.T) {
	resp := FraudResponse{Approved: true, FraudScore: 0.0}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var got FraudResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if got.Approved != true {
		t.Errorf("Approved = %v, want true", got.Approved)
	}
	if got.FraudScore != 0.0 {
		t.Errorf("FraudScore = %v, want 0.0", got.FraudScore)
	}
}

// --- MCC Risk Tests ---

func TestMCCRisk_KnownCodes(t *testing.T) {
	tests := []struct {
		mcc  string
		want float32
	}{
		{"5411", 0.15},
		{"5812", 0.30},
		{"5912", 0.20},
		{"5944", 0.45},
		{"7801", 0.80},
		{"7802", 0.75},
		{"7995", 0.85},
		{"4511", 0.35},
		{"5311", 0.25},
		{"5999", 0.50},
	}

	for _, tt := range tests {
		got := GetMCCRisk(tt.mcc)
		if got != tt.want {
			t.Errorf("GetMCCRisk(%q) = %v, want %v", tt.mcc, got, tt.want)
		}
	}
}

func TestMCCRisk_UnknownCode(t *testing.T) {
	unknowns := []string{"1234", "0000", "9999", ""}
	for _, mcc := range unknowns {
		got := GetMCCRisk(mcc)
		if got != MCCRiskDefault {
			t.Errorf("GetMCCRisk(%q) = %v, want default %v", mcc, got, MCCRiskDefault)
		}
	}
}

// --- Reference / Label Tests ---

func TestReferenceLabels(t *testing.T) {
	if LabelLegit != 0 {
		t.Errorf("LabelLegit = %v, want 0", LabelLegit)
	}
	if LabelFraud != 1 {
		t.Errorf("LabelFraud = %v, want 1", LabelFraud)
	}

	// 14 real dims + 2 padding zeros
	ref := Reference{
		Vector: [VectorDimsPad]float32{0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416, 0, 0},
		Label:  LabelLegit,
	}
	if ref.Label != LabelLegit {
		t.Errorf("ref.Label = %v, want LabelLegit (%v)", ref.Label, LabelLegit)
	}
	// Verify padding dims are zero
	if ref.Vector[14] != 0 || ref.Vector[15] != 0 {
		t.Errorf("padding dims should be 0, got [14]=%v [15]=%v", ref.Vector[14], ref.Vector[15])
	}
}

func TestVectorDimensions(t *testing.T) {
	if VectorDims != 14 {
		t.Errorf("VectorDims = %v, want 14", VectorDims)
	}
	if VectorDimsPad != 16 {
		t.Errorf("VectorDimsPad = %v, want 16", VectorDimsPad)
	}
}
