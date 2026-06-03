package internal

import (
	"time"
)

// Vectorize converte uma request em vetor de 16-dimensões (14 + 2 padding SIMD).
func Vectorize(req *FraudRequest) [VectorDimsPad]float32 {
	var v [VectorDimsPad]float32

	v[0] = clamp(float32(req.Transaction.Amount / MaxAmount))
	v[1] = clamp(float32(float64(req.Transaction.Installments) / MaxInstallments))
	v[2] = clamp(float32((req.Transaction.Amount / req.Customer.AvgAmount) / AmountVsAvgRatio))

	ts := parseTimestamp(req.Transaction.RequestedAt)
	v[3] = float32(ts.Hour()) / 23.0
	v[4] = float32(goWeekdayToSpec(ts.Weekday())) / 6.0

	// Dimensões 5 e 6: sentinela -1 se null
	if req.LastTransaction == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastTs := parseTimestamp(req.LastTransaction.Timestamp)
		minutes := ts.Sub(lastTs).Minutes()
		v[5] = clamp(float32(minutes / MaxMinutes))
		v[6] = clamp(float32(req.LastTransaction.KmFromCurrent / MaxKm))
	}

	v[7] = clamp(float32(req.Terminal.KmFromHome / MaxKm))
	v[8] = clamp(float32(float64(req.Customer.TxCount24h) / MaxTxCount24h))

	if req.Terminal.IsOnline { v[9] = 1 }
	if req.Terminal.CardPresent { v[10] = 1 }

	v[11] = 1
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			v[11] = 0
			break
		}
	}

	v[12] = GetMCCRisk(req.Merchant.MCC)
	v[13] = clamp(float32(req.Merchant.AvgAmount / MaxMerchantAvgAmount))

	// Os índices 14-15 do padding SIMD já estão zerados

	return v
}

// clamp mantém x entre 0.0 e 1.0.
func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// parseTimestamp extrai o objecto time.Time da string ISO.
func parseTimestamp(s string) time.Time {
	t, _ := time.Parse(time.RFC3339, s)
	return t
}

// goWeekdayToSpec converte o padrão de dia da semana do Go para o da spec (Seg=0).
func goWeekdayToSpec(wd time.Weekday) int {
	if wd == time.Sunday {
		return 6
	}
	return int(wd) - 1 // Mon=1→0, Tue=2→1, ..., Sat=6→5
}

// isoWeekday (apenas testes) extrai o dia da semana em número da string ISO.
func isoWeekday(s string) int {
	t := parseTimestamp(s)
	return goWeekdayToSpec(t.Weekday())
}
