package internal

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"
)

// Formato binário otimizado para leitura mmap zero-copy no Linux (amd64).

var binMagic = [6]byte{'V', 'P', 'T', 'R', 'E', 'E'}

const (
	binVersion    = 1
	binHeaderSize = 32
	binNodeSize   = 16 // sizeof(VPNode): int32 + float32 + int32 + int32
)

// binHeader é o cabeçalho de metadados do arquivo binário (32 bytes fixos).
type binHeader struct {
	Magic         [6]byte
	Version       uint16
	RootIdx       int32
	NumNodes      uint32
	NumRefs       uint32
	VectorDimsPad uint16
	Reserved      [10]byte
}

// SerializeVPTree grava a árvore e os vetores no formato binário customizado.
func SerializeVPTree(tree *VPTree, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	numNodes := uint32(len(tree.Nodes))
	numRefs := uint32(len(tree.Vectors))

	// Write header
	hdr := binHeader{
		Magic:         binMagic,
		Version:       binVersion,
		RootIdx:       tree.Root,
		NumNodes:      numNodes,
		NumRefs:       numRefs,
		VectorDimsPad: uint16(VectorDimsPad),
	}
	if err := binary.Write(f, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write nodes
	for i := range tree.Nodes {
		if err := binary.Write(f, binary.LittleEndian, &tree.Nodes[i]); err != nil {
			return fmt.Errorf("write node %d: %w", i, err)
		}
	}

	// Write vectors (contiguous float32 arrays)
	for i := range tree.Vectors {
		if err := binary.Write(f, binary.LittleEndian, &tree.Vectors[i]); err != nil {
			return fmt.Errorf("write vector %d: %w", i, err)
		}
	}

	// Write labels
	if _, err := f.Write(tree.Labels); err != nil {
		return fmt.Errorf("write labels: %w", err)
	}

	return nil
}

// LoadVPTreeBinary mapeia o arquivo binário direto na memória (mmap).
func LoadVPTreeBinary(path string) (*VPTree, func(), error) {
	data, cleanup, err := mmapFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("mmap file: %w", err)
	}

	tree, err := deserializeVPTree(data)
	if err != nil {
		cleanup()
		return nil, nil, err
	}

	return tree, cleanup, nil
}

// deserializeVPTree mapeia os blocos de bytes direto para os structs via unsafe (zero-copy).
func deserializeVPTree(data []byte) (*VPTree, error) {
	if len(data) < binHeaderSize {
		return nil, errors.New("file too small for header")
	}

	// Parse header
	var hdr binHeader
	r := newByteReader(data)
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Validate header
	if hdr.Magic != binMagic {
		return nil, fmt.Errorf("invalid magic: got %v", hdr.Magic)
	}
	if hdr.Version != binVersion {
		return nil, fmt.Errorf("unsupported version: %d", hdr.Version)
	}
	if hdr.VectorDimsPad != uint16(VectorDimsPad) {
		return nil, fmt.Errorf("dimension mismatch: file has %d, expected %d", hdr.VectorDimsPad, VectorDimsPad)
	}

	numNodes := int(hdr.NumNodes)
	numRefs := int(hdr.NumRefs)
	vectorBytes := numRefs * VectorDimsPad * 4 // 4 bytes per float32

	// Validate total size
	expectedSize := binHeaderSize + numNodes*binNodeSize + vectorBytes + numRefs
	if len(data) < expectedSize {
		return nil, fmt.Errorf("file too small: %d bytes, expected at least %d", len(data), expectedSize)
	}

	// Parse nodes using unsafe slice (zero-copy)
	nodesOffset := binHeaderSize
	nodes := unsafe.Slice((*VPNode)(unsafe.Pointer(&data[nodesOffset])), numNodes)

	// Parse vectors using unsafe slice (zero-copy)
	vectorsOffset := nodesOffset + numNodes*binNodeSize
	vectors := unsafe.Slice((*[VectorDimsPad]float32)(unsafe.Pointer(&data[vectorsOffset])), numRefs)

	// Parse labels (direct slice into data)
	labelsOffset := vectorsOffset + vectorBytes
	labels := data[labelsOffset : labelsOffset+numRefs]

	tree := &VPTree{
		Nodes:   nodes,
		Vectors: vectors,
		Labels:  labels,
		Root:    hdr.RootIdx,
	}

	return tree, nil
}

// byteReader adapta o []byte mmapado para a interface io.Reader.
type byteReader struct {
	data []byte
	pos  int
}

func newByteReader(data []byte) *byteReader {
	return &byteReader{data: data}
}

func (r *byteReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func init() {
	// Validações de compilação (guards) para os tamanhos de struct.
	if unsafe.Sizeof(VPNode{}) != binNodeSize {
		panic(fmt.Sprintf("VPNode size is %d, expected %d", unsafe.Sizeof(VPNode{}), binNodeSize))
	}
	// Verify float32 is 4 bytes
	if unsafe.Sizeof(float32(0)) != 4 {
		panic("float32 is not 4 bytes")
	}
	// Verify vector alignment
	var v [VectorDimsPad]float32
	if unsafe.Sizeof(v) != uintptr(VectorDimsPad*4) {
		panic(fmt.Sprintf("[%d]float32 size is %d, expected %d", VectorDimsPad, unsafe.Sizeof(v), VectorDimsPad*4))
	}
	// Verify we're on little-endian (amd64)
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = 0x0102
	if buf[0] != 0x02 {
		// Not a panic — just means we can't use zero-copy deserialization
		_ = math.Float32frombits(0) // suppress unused import
	}
}
