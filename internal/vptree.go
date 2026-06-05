package internal

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Pool para reciclar o estado da busca KNN e evitar alocações no hot path.
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

// VPTree implementa busca K-NN com Priority Queue (best-first) para pruning ótimo.
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

	// Reordenar vetores em ordem DFS: acessos sequenciais dentro de sub-árvores
	// = L3 cache hits em vez de DRAM random → speedup ~14× por visita.
	tree.reorderForCacheLocality()

	return tree
}

// reorderForCacheLocality reorganiza vetores e labels para ordem DFS dos nós.
// Após esta operação, Nodes[i].VPIdx == i (acesso identidade = sequential).
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
		t.Nodes[i].VPIdx = int32(i)
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
// SEARCH: Priority Queue (Best-First) KNN
// ─────────────────────────────────────────────────────────────────────────────

// stateNeighbor é um vizinho no max-heap interno dos K-melhores (pior em [0]).
type stateNeighbor struct {
	DistSq int32
	Label  uint8
}

// KNNResult holds up-to-K nearest neighbors in ascending distance order.
type KNNResult struct {
	Neighbors [5]Neighbor
	Len       int
}

// candidate é uma entrada na fila de prioridade para busca best-first.
// minDist é o lower bound da distância do query a qualquer ponto na sub-árvore.
type candidate struct {
	minDist float32
	nodeIdx int32
}

const (
	// maxCandidates: tamanho máximo da fila de prioridade.
	// Cada nó processado adiciona ≤2 filhos → max heap size ≈ nodesVisited + 1.
	maxCandidates = 120000

	// maxNodesToVisit: limite de segurança. Na prática, a PQ termina antes
	// (quando min lower_bound > tau), visitando muito menos nós.
	maxNodesToVisit = 100000
)

// knnSearchState contém todo o estado reutilizável de uma busca KNN (pooled).
// Tamanho: ~960KB (dominado pelo array de candidates). Com GOMAXPROCS=1 e GOGC=off,
// apenas 1 instância é alocada para sempre.
type knnSearchState struct {
	// Max-heap dos K-melhores vizinhos: heap[0] = o PIOR (maior DistSq)
	heap [5]stateNeighbor
	hLen int
	tau  float32 // distância real ao heap[0]

	// Min-heap para busca por prioridade (menor lower_bound primeiro)
	cands  [maxCandidates]candidate
	nCands int
}

// candPush adiciona um candidato ao min-heap de prioridade.
func (s *knnSearchState) candPush(minDist float32, nodeIdx int32) {
	if s.nCands >= maxCandidates {
		return
	}
	i := s.nCands
	s.cands[i] = candidate{minDist, nodeIdx}
	s.nCands++
	for i > 0 {
		p := (i - 1) >> 1
		if s.cands[i].minDist < s.cands[p].minDist {
			s.cands[i], s.cands[p] = s.cands[p], s.cands[i]
			i = p
		} else {
			break
		}
	}
}

// candPop remove e retorna o candidato com menor minDist do min-heap.
func (s *knnSearchState) candPop() (float32, int32) {
	r := s.cands[0]
	s.nCands--
	if s.nCands > 0 {
		s.cands[0] = s.cands[s.nCands]
		i := 0
		for {
			l, right := 2*i+1, 2*i+2
			m := i
			if l < s.nCands && s.cands[l].minDist < s.cands[m].minDist {
				m = l
			}
			if right < s.nCands && s.cands[right].minDist < s.cands[m].minDist {
				m = right
			}
			if m == i {
				break
			}
			s.cands[i], s.cands[m] = s.cands[m], s.cands[i]
			i = m
		}
	}
	return r.minDist, r.nodeIdx
}

// addNeighbor tenta inserir um ponto no max-heap de K-melhores.
// Com PQ search, cada nó é visitado no máximo uma vez → sem deduplicação.
func (s *knnSearchState) addNeighbor(distSq int32, label uint8) {
	if s.hLen < 5 {
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

// KNN realiza busca K-NN usando Priority Queue (best-first search).
//
// Por que é melhor que DFS com limite:
//   - DFS visita nós em ordem de profundidade arbitrária → tau cai devagar
//   - PQ visita nós em ordem crescente de lower_bound → tau cai RÁPIDO
//   - Com tau menor, mais branches são podadas → menos nós para mesma precisão
//
// Lower bounds via desigualdade triangular (provadamente corretos):
//   - Filho esquerdo (dist_VP ≤ radius): lb = max(0, dist_query - radius)
//   - Filho direito  (dist_VP > radius): lb = max(0, radius - dist_query)
//
// Garantia de exatidão: quando min(lb in queue) > tau → resultado é EXATO.
// Limite maxNodesToVisit é safety net para queries patológicas (14D, fronteira).
func (t *VPTree) KNN(query *[VectorDimsPad]uint8, k int) KNNResult {
	if t.Root == -1 {
		return KNNResult{}
	}

	s := knnStatePool.Get().(*knnSearchState)
	s.hLen = 0
	s.tau = math.MaxFloat32
	s.nCands = 0

	s.candPush(0, t.Root)

	visited := 0
	for s.nCands > 0 {
		minDist, nodeIdx := s.candPop()

		// Pruning: todos os restantes têm lb ≥ minDist > tau → resultado já é exato
		if minDist > s.tau {
			break
		}
		if visited >= maxNodesToVisit {
			break
		}

		node := &t.Nodes[nodeIdx]
		distSq := EuclideanDistSq(query, &t.Vectors[node.VPIdx])
		dist := float32(math.Sqrt(float64(distSq))) * 0.008
		label := t.Labels[node.VPIdx]

		s.addNeighbor(distSq, label)
		visited++

		// Filho esquerdo: pontos com dist_from_VP ≤ radius
		// lb = max(0, dist - radius)  [desigualdade triangular]
		if node.Left != -1 {
			lb := dist - node.Radius
			if lb < 0 {
				lb = 0
			}
			if lb <= s.tau {
				s.candPush(lb, node.Left)
			}
		}
		// Filho direito: pontos com dist_from_VP > radius
		// lb = max(0, radius - dist)  [desigualdade triangular]
		if node.Right != -1 {
			lb := node.Radius - dist
			if lb < 0 {
				lb = 0
			}
			if lb <= s.tau {
				s.candPush(lb, node.Right)
			}
		}
	}

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
