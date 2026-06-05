package internal

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Pool para reciclar o estado da busca KNN e evitar alocações no hot path.
// Com DFS search e sem array de candidatos, o estado é apenas ~80 bytes.
// Completamente seguro mesmo com centenas de goroutines concorrentes.
var knnStatePool = sync.Pool{
	New: func() interface{} {
		return &knnSearchState{}
	},
}

// VPNode é um nó da VP-Tree (16 bytes, ideal para mmap binário).
type VPNode struct {
	VPIdx  int32   // Índice do vetor de referência
	Radius float32 // Distância mediana até os filhos
	Left   int32   // Nó esquerdo (distância <= Radius)
	Right  int32   // Nó direito (distância > Radius)
}

// VPTree implementa busca K-NN com DFS e poda por desigualdade triangular.
// Vetores são reordenados em DFS-order para cache locality (~14× speedup).
type VPTree struct {
	Nodes   []VPNode
	Vectors [][VectorDimsPad]uint8
	Labels  []uint8
	Root    int32
}

// BuildVPTree constrói a árvore a partir dos vetores de referência.
func BuildVPTree(refs []Reference) *VPTree {
	if len(refs) == 0 {
		return &VPTree{Root: -1}
	}

	vectors := make([][VectorDimsPad]uint8, len(refs))
	labels := make([]uint8, len(refs))
	for i := range refs {
		vectors[i] = refs[i].Vector
		labels[i] = refs[i].Label
	}

	indices := make([]int, len(refs))
	for i := range indices {
		indices[i] = i
	}

	tree := &VPTree{
		Vectors: vectors,
		Labels:  labels,
		Nodes:   make([]VPNode, 0, len(refs)),
	}

	rng := rand.New(rand.NewSource(42))
	tree.Root = tree.buildRecursive(indices, rng)

	// Reordenar vetores em ordem DFS da árvore.
	// Resultado: quando DFS visita nodes sequencialmente, os vetores
	// correspondentes estão em memória contígua → L3 cache hits (~5ns)
	// em vez de DRAM random (~100ns) → speedup ~14×.
	tree.reorderForCacheLocality()

	return tree
}

// reorderForCacheLocality reorganiza vetores e labels para ordem DFS dos nós.
// Após: Nodes[i].VPIdx == i (identidade), e acesso durante busca é sequencial.
func (t *VPTree) reorderForCacheLocality() {
	n := len(t.Nodes)
	if n == 0 {
		return
	}
	newVectors := make([][VectorDimsPad]uint8, n)
	newLabels := make([]uint8, n)
	for i := range t.Nodes {
		vpIdx := t.Nodes[i].VPIdx
		newVectors[i] = t.Vectors[vpIdx]
		newLabels[i] = t.Labels[vpIdx]
		t.Nodes[i].VPIdx = int32(i) // identidade: VPIdx == node index
	}
	t.Vectors = newVectors
	t.Labels = newLabels
}

// buildRecursive cria os nós da árvore recursivamente.
func (t *VPTree) buildRecursive(indices []int, rng *rand.Rand) int32 {
	if len(indices) == 0 {
		return -1
	}

	if len(indices) == 1 {
		nodeIdx := int32(len(t.Nodes))
		t.Nodes = append(t.Nodes, VPNode{
			VPIdx:  int32(indices[0]),
			Radius: 0,
			Left:   -1,
			Right:  -1,
		})
		return nodeIdx
	}

	vpLocalIdx := t.selectVantagePoint(indices, rng)
	indices[0], indices[vpLocalIdx] = indices[vpLocalIdx], indices[0]
	vpRefIdx := indices[0]

	remaining := indices[1:]
	dists := make([]float32, len(remaining))
	for i, refIdx := range remaining {
		dists[i] = euclideanDist(&t.Vectors[vpRefIdx], &t.Vectors[refIdx])
	}

	medianDist := medianPartition(remaining, dists)

	mid := len(remaining) / 2
	leftIndices := remaining[:mid]
	rightIndices := remaining[mid:]

	nodeIdx := int32(len(t.Nodes))
	t.Nodes = append(t.Nodes, VPNode{
		VPIdx:  int32(vpRefIdx),
		Radius: medianDist,
	})

	left := t.buildRecursive(leftIndices, rng)
	right := t.buildRecursive(rightIndices, rng)
	t.Nodes[nodeIdx].Left = left
	t.Nodes[nodeIdx].Right = right

	return nodeIdx
}

// selectVantagePoint usa amostragem aleatória para achar o ponto com maior variância de distância.
func (t *VPTree) selectVantagePoint(indices []int, rng *rand.Rand) int {
	if len(indices) <= 5 {
		return rng.Intn(len(indices))
	}

	candidates := min(5, len(indices))
	sampleSize := min(20, len(indices))

	bestIdx := 0
	bestSpread := float32(-1)

	for c := 0; c < candidates; c++ {
		candidateLocalIdx := rng.Intn(len(indices))
		candidateRefIdx := indices[candidateLocalIdx]

		var sumDist, sumDistSq float32
		for s := 0; s < sampleSize; s++ {
			sampleLocalIdx := rng.Intn(len(indices))
			d := euclideanDist(&t.Vectors[candidateRefIdx], &t.Vectors[indices[sampleLocalIdx]])
			sumDist += d
			sumDistSq += d * d
		}

		mean := sumDist / float32(sampleSize)
		variance := sumDistSq/float32(sampleSize) - mean*mean
		if variance > bestSpread {
			bestSpread = variance
			bestIdx = candidateLocalIdx
		}
	}

	return bestIdx
}

// medianPartition separa os índices baseados na distância mediana.
func medianPartition(indices []int, dists []float32) float32 {
	n := len(indices)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return dists[0]
	}

	type idxDist struct {
		idx  int
		dist float32
	}
	pairs := make([]idxDist, n)
	for i := range pairs {
		pairs[i] = idxDist{idx: indices[i], dist: dists[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].dist < pairs[j].dist
	})

	for i := range pairs {
		indices[i] = pairs[i].idx
		dists[i] = pairs[i].dist
	}

	mid := n / 2
	return dists[mid-1]
}

// ─────────────────────────────────────────────────────────────────────────────
// SEARCH: DFS com poda por desigualdade triangular + DFS cache locality
// ─────────────────────────────────────────────────────────────────────────────

// stateNeighbor é um vizinho no max-heap interno dos K-melhores (pior em [0]).
// DFS visita cada nó exatamente uma vez → sem necessidade de VPIdx (dedup).
type stateNeighbor struct {
	DistSq int32
	Label  uint8
}

// KNNResult holds up-to-K nearest neighbors in ascending distance order.
type KNNResult struct {
	Neighbors [5]Neighbor
	Len       int
}

// maxNodesToVisit: ponto ótimo de score para CPU de 0.45 core por instância.
//
// Com vetores em DFS-order (cache-friendly, ~20ns por nó):
//   - 5000 nós × 20ns = 100μs de CPU por query
//   - wall clock = 100μs / 0.45 = 222μs
//   - ρ = 450 req/s × 222μs = 0.10 → muito leve
//   - p99 ≈ 0.5-1ms → p99_score = 3000 (máximo)
//
// E esperado ≈ 1400-1500: entre v3 (3K nós, E=1599) e v4 (50K nós, E=384).
const maxNodesToVisit = 5000

// knnSearchState é reciclável via sync.Pool — 80 bytes, completamente seguro.
// Sem array de candidatos (como o PQ erroneamente tinha): sem risco de OOM.
type knnSearchState struct {
	// K-melhores vizinhos: max-heap com o PIOR (maior DistSq) em heap[0]
	heap [5]stateNeighbor
	hLen int
	tau  float32 // distância real ao heap[0] (pior dos K-melhores)

	visited int // nós visitados nesta busca
}

// addNeighbor tenta inserir um ponto no max-heap de K-melhores.
// DFS garante que cada nó é visitado no máximo uma vez → sem deduplicação.
func (s *knnSearchState) addNeighbor(distSq int32, label uint8) {
	if s.hLen < 5 {
		// Heap não cheio: insere e faz sift-up (max-heap)
		s.heap[s.hLen] = stateNeighbor{DistSq: distSq, Label: label}
		s.hLen++
		i := s.hLen - 1
		for i > 0 {
			p := (i - 1) >> 1
			if s.heap[i].DistSq > s.heap[p].DistSq {
				s.heap[i], s.heap[p] = s.heap[p], s.heap[i]
				i = p
			} else {
				break
			}
		}
		if s.hLen == 5 {
			s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq))) * 0.008
		}
	} else if distSq < s.heap[0].DistSq {
		// Heap cheio: substitui o pior e faz sift-down (max-heap)
		s.heap[0] = stateNeighbor{DistSq: distSq, Label: label}
		i := 0
		for {
			l, r := 2*i+1, 2*i+2
			m := i
			if l < 5 && s.heap[l].DistSq > s.heap[m].DistSq {
				m = l
			}
			if r < 5 && s.heap[r].DistSq > s.heap[m].DistSq {
				m = r
			}
			if m == i {
				break
			}
			s.heap[i], s.heap[m] = s.heap[m], s.heap[i]
			i = m
		}
		s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq))) * 0.008
	}
}

// dfsSearch realiza DFS recursivo com poda por desigualdade triangular.
//
// Poda: se a distância mínima possível de qualquer ponto no subtree > tau,
// o subtree inteiro pode ser ignorado. Isso é garantido por:
//   - dist < radius → subtree direito: dist mínima ≥ radius - dist
//   - dist ≥ radius → subtree esquerdo: dist mínima ≥ dist - radius
//
// Com DFS-order nos vetores, a travessia greedy (root → leaf) acessa
// memória sequencialmente → L3 cache hits → alta eficiência.
func (t *VPTree) dfsSearch(s *knnSearchState, query *[VectorDimsPad]uint8, nodeIdx int32) {
	if nodeIdx == -1 || s.visited >= maxNodesToVisit {
		return
	}

	node := &t.Nodes[nodeIdx]
	distSq := EuclideanDistSq(query, &t.Vectors[node.VPIdx])
	dist := float32(math.Sqrt(float64(distSq))) * 0.008

	s.addNeighbor(distSq, t.Labels[node.VPIdx])
	s.visited++

	if dist < node.Radius {
		// Query dentro da bola: vai para o lado esquerdo primeiro (mais promissor)
		t.dfsSearch(s, query, node.Left)
		// Só vai para o direito se o raio de busca ultrapassa a fronteira
		if node.Radius-dist <= s.tau {
			t.dfsSearch(s, query, node.Right)
		}
	} else {
		// Query fora da bola: vai para o lado direito primeiro (mais promissor)
		t.dfsSearch(s, query, node.Right)
		// Só vai para o esquerdo se o raio de busca ultrapassa a fronteira
		if dist-node.Radius <= s.tau {
			t.dfsSearch(s, query, node.Left)
		}
	}
}

// KNN encontra os K vizinhos mais próximos usando DFS com poda triangular.
//
// O DFS naturalmente visita o "caminho guloso" (root → leaf) primeiro,
// inicializando tau rapidamente com bons vizinhos. Isso maximiza o pruning
// das branches subsequentes, mantendo a busca eficiente.
//
// Com reorderForCacheLocality: vetores no caminho guloso estão em memória
// contígua → L1/L2 cache hits nos primeiros 22 nós → tau ótimo inicial.
func (t *VPTree) KNN(query *[VectorDimsPad]uint8, k int) KNNResult {
	if t.Root == -1 {
		return KNNResult{}
	}

	s := knnStatePool.Get().(*knnSearchState)
	s.hLen = 0
	s.tau = math.MaxFloat32
	s.visited = 0

	t.dfsSearch(s, query, t.Root)

	var result KNNResult
	result.Len = s.hLen
	for i := 0; i < s.hLen; i++ {
		result.Neighbors[i] = Neighbor{
			DistSq: float32(s.heap[i].DistSq) * 0.000064,
			Label:  s.heap[i].Label,
		}
	}
	knnStatePool.Put(s)

	// Ordenar por distância crescente (insertion sort, ótimo para ≤5 elementos)
	for i := 1; i < result.Len; i++ {
		key := result.Neighbors[i]
		j := i - 1
		for j >= 0 && result.Neighbors[j].DistSq > key.DistSq {
			result.Neighbors[j+1] = result.Neighbors[j]
			j--
		}
		result.Neighbors[j+1] = key
	}

	return result
}

// euclideanDist calcula distância real entre dois vetores quantizados (usado no build).
func euclideanDist(a, b *[VectorDimsPad]uint8) float32 {
	return float32(math.Sqrt(float64(EuclideanDistSq(a, b)))) * 0.008
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Warmup toca todas as páginas de memória para eliminar page faults no hot path.
func (t *VPTree) Warmup() {
	var sum uint32
	for i := range t.Vectors {
		for d := 0; d < VectorDimsPad; d++ {
			sum += uint32(t.Vectors[i][d])
		}
	}
	for i := range t.Nodes {
		sum += uint32(t.Nodes[i].VPIdx)
		sum += uint32(t.Nodes[i].Left)
		sum += uint32(t.Nodes[i].Right)
	}
	for i := range t.Labels {
		sum += uint32(t.Labels[i])
	}
	if sum == 0xdeadbeef {
		println(sum)
	}
}
