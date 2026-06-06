package internal

const BFDims = 14 // actual dimensions (no padding needed)

// BFDataset holds the reference data in SoA layout for cache-friendly scanning.
type BFDataset struct {
	NumRefs int
	// SoA: Dims[d][i] = dimension d of reference i, quantized to int16
	Dims   [BFDims][]int16
	Labels []uint8
}

// bfHeapEntry is a neighbor tracked in the max-heap during brute-force KNN.
type bfHeapEntry struct {
	distSq int64
	label  uint8
}

// BruteForceKNN5 finds the 5 nearest neighbors by scanning ALL references.
// query is the query vector quantized to int16[14].
// Returns the fraud count (0-5) directly, avoiding allocations entirely.
func BruteForceKNN5(query *[BFDims]int16, ds *BFDataset) int {
	var heap [5]bfHeapEntry
	hLen := 0
	worstDist := int64(0x7FFFFFFFFFFFFFFF)

	for i := 0; i < ds.NumRefs; i++ {
		var distSq int64
		for d := 0; d < BFDims; d++ {
			diff := int64(query[d]) - int64(ds.Dims[d][i])
			distSq += diff * diff
		}

		if hLen < 5 {
			// Heap not full: insert
			heap[hLen] = bfHeapEntry{distSq: distSq, label: ds.Labels[i]}
			hLen++
			// Sift up
			j := hLen - 1
			for j > 0 {
				p := (j - 1) >> 1
				if heap[j].distSq > heap[p].distSq {
					heap[j], heap[p] = heap[p], heap[j]
					j = p
				} else {
					break
				}
			}
			if hLen == 5 {
				worstDist = heap[0].distSq
			}
		} else if distSq < worstDist {
			// Replace worst
			heap[0] = bfHeapEntry{distSq: distSq, label: ds.Labels[i]}
			// Sift down
			j := 0
			for {
				l, r := 2*j+1, 2*j+2
				m := j
				if l < 5 && heap[l].distSq > heap[m].distSq {
					m = l
				}
				if r < 5 && heap[r].distSq > heap[m].distSq {
					m = r
				}
				if m == j {
					break
				}
				heap[j], heap[m] = heap[m], heap[j]
				j = m
			}
			worstDist = heap[0].distSq
		}
	}

	// Count fraud labels
	fraudCount := 0
	for i := 0; i < hLen; i++ {
		if heap[i].label != 0 {
			fraudCount++
		}
	}
	return fraudCount
}

// QuantizeVecToInt16 converts a float32 vector to int16 for brute-force KNN.
// Maps [-1, 1] → [-10000, 10000].
func QuantizeVecToInt16(vec [VectorDimsPad]float32) [BFDims]int16 {
	var q [BFDims]int16
	for i := 0; i < BFDims; i++ {
		v := vec[i]
		iv := int32(v * 10000)
		if iv > 32767 {
			iv = 32767
		}
		if iv < -32768 {
			iv = -32768
		}
		q[i] = int16(iv)
	}
	return q
}

// QuantizeFloat32ToInt16 converts a single float32 value in [-1, 1] to int16.
// Maps [-1, 1] → [-10000, 10000].
func QuantizeFloat32ToInt16(f float32) int16 {
	iv := int32(f * 10000)
	if iv > 32767 {
		iv = 32767
	}
	if iv < -32768 {
		iv = -32768
	}
	return int16(iv)
}

// Warmup touches all memory pages to fault them in before serving requests.
func (ds *BFDataset) Warmup() {
	for d := 0; d < BFDims; d++ {
		for i := 0; i < len(ds.Dims[d]); i += 4096 / 2 { // 4096 bytes / 2 bytes per int16
			_ = ds.Dims[d][i]
		}
	}
	for i := 0; i < len(ds.Labels); i += 4096 {
		_ = ds.Labels[i]
	}
}
