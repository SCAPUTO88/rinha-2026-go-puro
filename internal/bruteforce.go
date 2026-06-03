package internal

// Neighbor representa um nó na avaliação do KNN.
type Neighbor struct {
	DistSq float32 // Squared Euclidean distance to the query vector
	Label  uint8   // 0 = legit, 1 = fraud
}

// BruteForceKNN varre todos os vetores para achar os K vizinhos.
// Usado apenas nos testes como baseline exato O(N).
func BruteForceKNN(query *[VectorDimsPad]float32, refs []Reference, k int) []Neighbor {
	if k > len(refs) {
		k = len(refs)
	}

	// Max-heap simples com capacidade K.
	heap := make([]Neighbor, 0, k)

	for i := range refs {
		dist := EuclideanDistSq(query, &refs[i].Vector)

		if len(heap) < k {
			// Heap not full yet — insert and maintain sorted order
			heap = insertSorted(heap, Neighbor{DistSq: dist, Label: refs[i].Label})
		} else if dist < heap[k-1].DistSq {
			// Closer than the farthest in heap — replace and re-sort
			heap[k-1] = Neighbor{DistSq: dist, Label: refs[i].Label}
			heap = insertSorted(heap[:k-1], heap[k-1])
		}
	}

	return heap
}

// insertSorted insere no slice mantendo ordenado por DistSq.
func insertSorted(sorted []Neighbor, n Neighbor) []Neighbor {
	i := len(sorted)
	sorted = append(sorted, n)
	for i > 0 && sorted[i].DistSq < sorted[i-1].DistSq {
		sorted[i], sorted[i-1] = sorted[i-1], sorted[i]
		i--
	}
	return sorted
}
