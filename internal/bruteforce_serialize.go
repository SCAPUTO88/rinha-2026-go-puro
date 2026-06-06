package internal

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"unsafe"
)

// Binary format for BFDataset:
//
// Header (32 bytes):
//   Magic: "BFKNN\x00" (6 bytes)
//   Version: uint16 = 1
//   NumRefs: uint32
//   NumDims: uint16 = 14
//   Reserved: 18 bytes
//
// Dim 0 data: NumRefs × int16 (little-endian)
// Dim 1 data: NumRefs × int16
// ...
// Dim 13 data: NumRefs × int16
// Labels: NumRefs × uint8

var bfMagic = [6]byte{'B', 'F', 'K', 'N', 'N', 0}

const (
	bfVersion    = 1
	bfHeaderSize = 32
)

type bfHeader struct {
	Magic    [6]byte
	Version  uint16
	NumRefs  uint32
	NumDims  uint16
	Reserved [18]byte
}

// SerializeBFDataset writes a BFDataset to a binary file.
func SerializeBFDataset(ds *BFDataset, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	hdr := bfHeader{
		Magic:   bfMagic,
		Version: bfVersion,
		NumRefs: uint32(ds.NumRefs),
		NumDims: uint16(BFDims),
	}
	if err := binary.Write(f, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write each dimension's data
	for d := 0; d < BFDims; d++ {
		if err := binary.Write(f, binary.LittleEndian, ds.Dims[d]); err != nil {
			return fmt.Errorf("write dim %d: %w", d, err)
		}
	}

	// Write labels
	if _, err := f.Write(ds.Labels); err != nil {
		return fmt.Errorf("write labels: %w", err)
	}

	return nil
}

// LoadBFDataset loads a BFDataset from a binary file.
// On Linux it uses mmap for zero-copy; on Windows it uses ReadFile.
// Returns the dataset, a cleanup function, and any error.
func LoadBFDataset(path string) (*BFDataset, func(), error) {
	data, cleanup, err := mmapFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("load file: %w", err)
	}

	ds, err := deserializeBFDataset(data)
	if err != nil {
		cleanup()
		return nil, nil, err
	}

	return ds, cleanup, nil
}

// deserializeBFDataset maps the binary data into a BFDataset using zero-copy where possible.
func deserializeBFDataset(data []byte) (*BFDataset, error) {
	if len(data) < bfHeaderSize {
		return nil, errors.New("file too small for header")
	}

	// Parse header
	var hdr bfHeader
	r := newByteReader(data)
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Validate
	if hdr.Magic != bfMagic {
		return nil, fmt.Errorf("invalid magic: got %v, expected BFKNN", hdr.Magic)
	}
	if hdr.Version != bfVersion {
		return nil, fmt.Errorf("unsupported version: %d", hdr.Version)
	}
	if hdr.NumDims != uint16(BFDims) {
		return nil, fmt.Errorf("dimension mismatch: file has %d, expected %d", hdr.NumDims, BFDims)
	}

	numRefs := int(hdr.NumRefs)
	dimBytes := numRefs * 2 // 2 bytes per int16
	totalDimBytes := BFDims * dimBytes
	expectedSize := bfHeaderSize + totalDimBytes + numRefs
	if len(data) < expectedSize {
		return nil, fmt.Errorf("file too small: %d bytes, expected at least %d", len(data), expectedSize)
	}

	ds := &BFDataset{
		NumRefs: numRefs,
	}

	// Map each dimension's int16 slice using unsafe (zero-copy from mmap)
	offset := bfHeaderSize
	for d := 0; d < BFDims; d++ {
		ptr := unsafe.Pointer(&data[offset])
		ds.Dims[d] = unsafe.Slice((*int16)(ptr), numRefs)
		offset += dimBytes
	}

	// Map labels (direct slice into data)
	ds.Labels = data[offset : offset+numRefs]

	return ds, nil
}
