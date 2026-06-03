package internal

const (
	// FraudThreshold é o limite fixo para aprovar uma transação (da spec).
	FraudThreshold = 0.6
)

// ComputeFraudScore calcula a proporção de fraudes entre os vizinhos (0.0 a 1.0).
func ComputeFraudScore(neighbors []Neighbor) float32 {
	if len(neighbors) == 0 {
		return 0
	}
	fraudCount := 0
	for _, n := range neighbors {
		if n.Label == LabelFraud {
			fraudCount++
		}
	}
	return float32(fraudCount) / float32(len(neighbors))
}

// IsApproved retorna true se o score for menor que o threshold (0.6).
func IsApproved(fraudScore float32) bool {
	return fraudScore < FraudThreshold
}
