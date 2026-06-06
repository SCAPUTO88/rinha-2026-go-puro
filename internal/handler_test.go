package internal

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestHandler creates a handler backed by a small VP-Tree for testing.
func newTestHandler(t *testing.T) *FraudHandler {
	t.Helper()
	ds := smallBFDataset()
	return NewFraudHandler(ds)
}

func TestReadyHandler_Returns200(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	req := httptest.NewRequest("GET", "/ready", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("GET /ready status = %d, want 200", w.Code)
	}
}

func TestFraudScoreHandler_ValidPayload_ReturnsJSON(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

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

	req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	t.Logf("response: %s", body)

	// Must contain "approved" and "fraud_score"
	if !strings.Contains(body, `"approved"`) {
		t.Error("response missing 'approved' field")
	}
	if !strings.Contains(body, `"fraud_score"`) {
		t.Error("response missing 'fraud_score' field")
	}

	// Content-Type must be application/json
	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestFraudScoreHandler_NullLastTransaction(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	payload := `{
		"id": "tx-test",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T12:00:00Z"},
		"customer": {"avg_amount": 50, "tx_count_24h": 1, "known_merchants": []},
		"merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 100},
		"terminal": {"is_online": true, "card_present": false, "km_from_home": 10},
		"last_transaction": null
	}`

	req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader(payload))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (no crash on null last_transaction)", w.Code)
	}
}

func TestFraudScoreHandler_InvalidJSON_Fallback(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	// Fallback strategy: return approved=true, fraud_score=0.0 on error
	// Better a false positive (weight=1) than an HTTP error (weight=5)
	req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader("{invalid json"))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (fallback on invalid JSON)", w.Code)
	}

	body := w.Body.String()
	// Should default to approved
	if !strings.Contains(body, `"approved":true`) {
		t.Errorf("fallback should approve, got: %s", body)
	}
	if !strings.Contains(body, `"fraud_score":0`) {
		t.Errorf("fallback should have fraud_score 0, got: %s", body)
	}
}

func TestFraudScoreHandler_EmptyBody_Fallback(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	req := httptest.NewRequest("POST", "/fraud-score", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (fallback on empty body)", w.Code)
	}
}

func TestFraudScoreHandler_ResponseFormat(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	// Use the fraud transaction — should get high fraud_score
	payload := `{
		"id": "tx-3330991687",
		"transaction": {"amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
		"customer": {"avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"]},
		"merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 952.2745933273},
		"last_transaction": null
	}`

	req := httptest.NewRequest("POST", "/fraud-score", strings.NewReader(payload))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	body := w.Body.String()
	t.Logf("fraud response: %s", body)

	// fraud_score should be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
	validScores := []string{
		`"fraud_score":0.0`, `"fraud_score":0.2`, `"fraud_score":0.4`,
		`"fraud_score":0.6`, `"fraud_score":0.8`, `"fraud_score":1.0`,
		// Also accept without trailing zero
		`"fraud_score":0`, `"fraud_score":1`,
	}
	found := false
	for _, vs := range validScores {
		if strings.Contains(body, vs) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("fraud_score not a valid value, body: %s", body)
	}
}

func TestFraudScoreHandler_MethodNotAllowed(t *testing.T) {
	handler := newTestHandler(t)
	mux := handler.RegisterRoutes()

	// GET on /fraud-score should still not crash
	req := httptest.NewRequest("GET", "/fraud-score", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	// We don't care about the status code, just that it doesn't panic
	_ = w.Code

	// POST on /ready should also not crash
	req2 := httptest.NewRequest("POST", "/ready", io.NopCloser(strings.NewReader("")))
	w2 := httptest.NewRecorder()
	mux.ServeHTTP(w2, req2)
	_ = w2.Code
}
