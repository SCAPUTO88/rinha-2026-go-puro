package internal

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
)

// rawReference mapeia a estrutura original do JSON.
type rawReference struct {
	Vector [VectorDims]float64 `json:"vector"` // JSON has 14 dims
	Label  string              `json:"label"`  // "legit" or "fraud"
}

// LoadReferencesJSON carrega o dataset JSON de disco.
func LoadReferencesJSON(path string) ([]Reference, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return decodeReferences(f)
}

// LoadReferencesGzip carrega o dataset JSON compactado via gzip.
func LoadReferencesGzip(path string) ([]Reference, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	return decodeReferences(gz)
}

// decodeReferences faz o parse em stream (token a token) para não estourar a RAM.
func decodeReferences(r io.Reader) ([]Reference, error) {
	dec := json.NewDecoder(r)

	// Consume the opening '[' of the array
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, err
	}

	// Pre-allocate for expected size (3M for full dataset, grows as needed)
	refs := make([]Reference, 0, 1024)

	for dec.More() {
		var raw rawReference
		if err := dec.Decode(&raw); err != nil {
			return nil, err
		}

		var ref Reference
		// Copy 14 dims from float64 (JSON) to float32 (internal), padding 14-15 stay zero
		for i := 0; i < VectorDims; i++ {
			ref.Vector[i] = float32(raw.Vector[i])
		}
		// ref.Vector[14] and [15] are already zero from struct initialization

		switch raw.Label {
		case "fraud":
			ref.Label = LabelFraud
		default: // "legit" or anything else
			ref.Label = LabelLegit
		}

		refs = append(refs, ref)
	}

	return refs, nil
}
