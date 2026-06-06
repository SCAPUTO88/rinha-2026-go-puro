package internal

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// LoadReferencesAsBFDataset loads references from a JSON or gzip file and
// builds a BFDataset with int16 quantization in SoA layout.
func LoadReferencesAsBFDataset(path string) (*BFDataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var reader io.Reader = f

	// Auto-detect gzip by file extension
	if len(path) > 3 && path[len(path)-3:] == ".gz" {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil, fmt.Errorf("gzip reader: %w", err)
		}
		defer gz.Close()
		reader = gz
	}

	return decodeBFDataset(reader)
}

// decodeBFDataset stream-decodes a JSON array of references into a BFDataset.
func decodeBFDataset(r io.Reader) (*BFDataset, error) {
	dec := json.NewDecoder(r)

	// Consume opening '['
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, fmt.Errorf("expected '[', got %v", tok)
	}

	// Pre-allocate for expected 3M references
	const initialCap = 3_000_000
	ds := &BFDataset{
		Labels: make([]uint8, 0, initialCap),
	}
	for d := 0; d < BFDims; d++ {
		ds.Dims[d] = make([]int16, 0, initialCap)
	}

	for dec.More() {
		var raw rawReference
		if err := dec.Decode(&raw); err != nil {
			return nil, fmt.Errorf("decode reference: %w", err)
		}

		// Quantize float64 vector to int16
		for d := 0; d < BFDims; d++ {
			ds.Dims[d] = append(ds.Dims[d], QuantizeFloat32ToInt16(float32(raw.Vector[d])))
		}

		// Label
		var label uint8
		if raw.Label == "fraud" {
			label = LabelFraud
		}
		ds.Labels = append(ds.Labels, label)
	}

	ds.NumRefs = len(ds.Labels)
	return ds, nil
}
