package internal

// FraudRequest é o payload JSON que a API recebe.
type FraudRequest struct {
	ID              string           `json:"id"`
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"` // pointer para suportar null
}

type Transaction struct {
	Amount      float64 `json:"amount"`
	Installments int    `json:"installments"`
	RequestedAt string  `json:"requested_at"` // ISO 8601 UTC
}

type Customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     string  `json:"timestamp"` // ISO 8601 UTC
	KmFromCurrent float64 `json:"km_from_current"`
}

// FraudResponse é a resposta JSON da API.
type FraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

// Dimensões do vetor: 14 da spec + 2 de padding para alinhamento SIMD na L1 cache.
const (
	VectorDims    = 14
	VectorDimsPad = 16
)

// Reference é um item do dataset, com vetor alinhado e sua classificação (0=legit, 1=fraud).
type Reference struct {
	Vector [VectorDimsPad]uint8
	Label  uint8
}

const (
	LabelLegit uint8 = 0
	LabelFraud uint8 = 1
)

// QuantizeFloat32 converte um valor float32 na faixa [-1.0, 1.0] para uint8.
// O valor sentinela -1.0 vira 255.
// Outros valores são limitados a [0.0, 1.0] e mapeados para [0, 250].
func QuantizeFloat32(val float32) uint8 {
	if val < -0.5 {
		return 255
	}
	if val < 0 {
		val = 0
	} else if val > 1 {
		val = 1
	}
	return uint8(val*250.0 + 0.5)
}

// DequantizeToFloat32 reconverte o uint8 quantizado de volta para float32.
func DequantizeToFloat32(val uint8) float32 {
	if val == 255 {
		return -1.0
	}
	return float32(val) / 250.0
}
