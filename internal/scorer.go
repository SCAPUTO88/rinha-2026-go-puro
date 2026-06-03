package internal

const (
	// FraudThreshold é o limite fixo para aprovar uma transação (da spec).
	FraudThreshold = 0.6
)

// ComputeFraudScore calcula a proporção de fraudes entre os vizinhos (0.0 a 1.0).
func ComputeFraudScore(res KNNResult) float32 {
	if res.Len == 0 {
		return 0
	}
	fraudCount := 0
	for i := 0; i < res.Len; i++ {
		if res.Neighbors[i].Label == LabelFraud {
			fraudCount++
		}
	}
	return float32(fraudCount) / float32(res.Len)
}

// IsApproved retorna true se o score for menor que o threshold (0.6).
func IsApproved(fraudScore float32) bool {
	return fraudScore < FraudThreshold
}
