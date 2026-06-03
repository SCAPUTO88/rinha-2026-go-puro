package internal

// Constantes de normalização definidas no dataset e fixas durante runtime.
const (
	MaxAmount           = 10000.0
	MaxInstallments     = 12.0
	AmountVsAvgRatio    = 10.0
	MaxMinutes          = 1440.0
	MaxKm               = 1000.0
	MaxTxCount24h       = 20.0
	MaxMerchantAvgAmount = 10000.0
)

// MCCRisk mapeia códigos MCC para seus scores de risco.
var MCCRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}

// MCCRiskDefault é o valor padrão para MCCs desconhecidos.
const MCCRiskDefault float32 = 0.5

// GetMCCRisk retorna o score de risco do MCC.
func GetMCCRisk(mcc string) float32 {
	if risk, ok := MCCRisk[mcc]; ok {
		return risk
	}
	return MCCRiskDefault
}
