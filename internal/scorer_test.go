package internal

import (
	"testing"
)

func TestComputeFraudScore_AllFraud(t *testing.T) {
	neighbors := []Neighbor{
		{Label: LabelFraud}, {Label: LabelFraud}, {Label: LabelFraud},
		{Label: LabelFraud}, {Label: LabelFraud},
	}
	got := ComputeFraudScore(neighbors)
	if got != 1.0 {
		t.Errorf("ComputeFraudScore(all fraud) = %v, want 1.0", got)
	}
}

func TestComputeFraudScore_AllLegit(t *testing.T) {
	neighbors := []Neighbor{
		{Label: LabelLegit}, {Label: LabelLegit}, {Label: LabelLegit},
		{Label: LabelLegit}, {Label: LabelLegit},
	}
	got := ComputeFraudScore(neighbors)
	if got != 0.0 {
		t.Errorf("ComputeFraudScore(all legit) = %v, want 0.0", got)
	}
}

func TestComputeFraudScore_Mixed3of5(t *testing.T) {
	neighbors := []Neighbor{
		{Label: LabelFraud}, {Label: LabelLegit}, {Label: LabelFraud},
		{Label: LabelLegit}, {Label: LabelFraud},
	}
	got := ComputeFraudScore(neighbors)
	if got != 0.6 {
		t.Errorf("ComputeFraudScore(3/5 fraud) = %v, want 0.6", got)
	}
}

func TestComputeFraudScore_Mixed2of5(t *testing.T) {
	neighbors := []Neighbor{
		{Label: LabelFraud}, {Label: LabelLegit}, {Label: LabelFraud},
		{Label: LabelLegit}, {Label: LabelLegit},
	}
	got := ComputeFraudScore(neighbors)
	if got != 0.4 {
		t.Errorf("ComputeFraudScore(2/5 fraud) = %v, want 0.4", got)
	}
}

func TestIsApproved_BelowThreshold(t *testing.T) {
	// 0.4 < 0.6 → approved
	if !IsApproved(0.4) {
		t.Error("IsApproved(0.4) should be true")
	}
}

func TestIsApproved_AtThreshold(t *testing.T) {
	// 0.6 is NOT < 0.6 → denied
	if IsApproved(0.6) {
		t.Error("IsApproved(0.6) should be false (threshold is strict <)")
	}
}

func TestIsApproved_AboveThreshold(t *testing.T) {
	if IsApproved(0.8) {
		t.Error("IsApproved(0.8) should be false")
	}
}

func TestIsApproved_Zero(t *testing.T) {
	if !IsApproved(0.0) {
		t.Error("IsApproved(0.0) should be true")
	}
}

func TestIsApproved_One(t *testing.T) {
	if IsApproved(1.0) {
		t.Error("IsApproved(1.0) should be false")
	}
}
