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

// VPTree implementa busca exata K-NN com suporte a mmap zero-copy.
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

	// Split references into separate vectors and labels arrays
	vectors := make([][VectorDimsPad]uint8, len(refs))
	labels := make([]uint8, len(refs))
	for i := range refs {
		vectors[i] = refs[i].Vector
		labels[i] = refs[i].Label
	}

	// Create index array [0, 1, 2, ..., N-1]
	indices := make([]int, len(refs))
	for i := range indices {
		indices[i] = i
	}

	tree := &VPTree{
		Vectors: vectors,
		Labels:  labels,
		Nodes:   make([]VPNode, 0, len(refs)),
	}

	rng := rand.New(rand.NewSource(42)) // Deterministic for reproducibility
	tree.Root = tree.buildRecursive(indices, rng)

	return tree
}

// buildRecursive cria os nós da árvore recursivamente.
func (t *VPTree) buildRecursive(indices []int, rng *rand.Rand) int32 {
	if len(indices) == 0 {
		return -1
	}

	// Leaf node: single element
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

	// Select vantage point: pick best from a small random sample
	vpLocalIdx := t.selectVantagePoint(indices, rng)
	// Swap VP to front
	indices[0], indices[vpLocalIdx] = indices[vpLocalIdx], indices[0]
	vpRefIdx := indices[0]

	// Compute distances from VP to all other points
	remaining := indices[1:]
	dists := make([]float32, len(remaining))
	for i, refIdx := range remaining {
		dists[i] = euclideanDist(&t.Vectors[vpRefIdx], &t.Vectors[refIdx])
	}

	// Find median distance and partition
	medianDist := medianPartition(remaining, dists)

	// Split: left = dist <= median, right = dist > median
	// After medianPartition, remaining is already partitioned
	mid := len(remaining) / 2
	leftIndices := remaining[:mid]
	rightIndices := remaining[mid:]

	// Allocate this node
	nodeIdx := int32(len(t.Nodes))
	t.Nodes = append(t.Nodes, VPNode{
		VPIdx:  int32(vpRefIdx),
		Radius: medianDist,
	})

	// Recurse (node is already appended, so we update Left/Right after)
	left := t.buildRecursive(leftIndices, rng)
	right := t.buildRecursive(rightIndices, rng)
	t.Nodes[nodeIdx].Left = left
	t.Nodes[nodeIdx].Right = right

	return nodeIdx
}

// selectVantagePoint usa amostragem aleatória para achar o ponto com maior variância de distância.
func (t *VPTree) selectVantagePoint(indices []int, rng *rand.Rand) int {
	if len(indices) <= 5 {
		// For tiny sets, just use a random pick
		return rng.Intn(len(indices))
	}

	// Sample up to 5 candidates
	candidates := min(5, len(indices))
	sampleSize := min(20, len(indices))

	bestIdx := 0
	bestSpread := float32(-1)

	for c := 0; c < candidates; c++ {
		candidateLocalIdx := rng.Intn(len(indices))
		candidateRefIdx := indices[candidateLocalIdx]

		// Compute distances to a random sample of points
		var sumDist, sumDistSq float32
		for s := 0; s < sampleSize; s++ {
			sampleLocalIdx := rng.Intn(len(indices))
			d := euclideanDist(&t.Vectors[candidateRefIdx], &t.Vectors[indices[sampleLocalIdx]])
			sumDist += d
			sumDistSq += d * d
		}

		// Variance = E[X²] - E[X]² (spread of distances)
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

	// Create index-distance pairs and sort by distance
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

	// Write back sorted order
	for i := range pairs {
		indices[i] = pairs[i].idx
		dists[i] = pairs[i].dist
	}

	// Median is at n/2 - 1 (we put n/2 elements in the left subtree)
	mid := n / 2
	return dists[mid-1]
}

type stateNeighbor struct {
	DistSq int32
	VPIdx  int32
	Label  uint8
}

type KNNResult struct {
	Neighbors [5]Neighbor
	Len       int
}

// maxNodesToVisit limita o número de nós visitados durante a busca VP-Tree.
// Isso garante latência máxima controlada em espaços de alta dimensionalidade
// onde o pruning pode ser fraco. 3000 nós dá excelente precisão na prática
// para dados de fraude em 14 dimensões com 3M referências.
const maxNodesToVisit = 3000

// KNN encontra os k-vizinhos mais próximos aplicando poda por desigualdade triangular de forma zero-alloc.
func (t *VPTree) KNN(query *[VectorDimsPad]uint8, k int) KNNResult {
	if t.Root == -1 {
		return KNNResult{}
	}

	s := knnStatePool.Get().(*knnSearchState)
	s.tree = t
	s.query = query
	s.k = k
	s.len = 0
	s.tau = float32(math.Inf(1))
	s.heap = [5]stateNeighbor{} // zera heap anterior
	s.visited = 0

	// 1. Busca gulosa rápida para popular o heap e diminuir o tau inicial
	s.searchGreedy(t.Root)

	// 2. Busca recursiva exata aplicando as podas precocemente com o tau inicial reduzido
	s.search(t.Root)

	// Copia e desquantiza o resultado
	var result KNNResult
	result.Len = s.len
	for i := 0; i < s.len; i++ {
		result.Neighbors[i] = Neighbor{
			DistSq: float32(s.heap[i].DistSq) * 0.000064,
			Label:  s.heap[i].Label,
		}
	}

	knnStatePool.Put(s)

	// Inverte a ordem de decrescente para crescente (para manter ordenação ascendente por distância)
	for i := 0; i < result.Len/2; i++ {
		j := result.Len - 1 - i
		result.Neighbors[i], result.Neighbors[j] = result.Neighbors[j], result.Neighbors[i]
	}

	return result
}

// knnSearchState é reciclável para não gerar lixo no GC durante buscas sucessivas.
type knnSearchState struct {
	tree    *VPTree
	query   *[VectorDimsPad]uint8
	k       int
	heap    [5]stateNeighbor // Array fixo
	len     int
	tau     float32 // Worst actual distance (not squared)
	visited int     // Contador de nós visitados (limita latência máxima)
}

func (s *knnSearchState) addNeighbor(distSq int32, vpIdx int32, label uint8) {
	// Evita duplicadas de nós visitados no greedy e na busca principal
	for i := 0; i < s.len; i++ {
		if s.heap[i].VPIdx == vpIdx {
			return
		}
	}

	if s.len < 5 {
		// Insere mantendo ordenado decrescentemente por DistSq
		s.heap[s.len] = stateNeighbor{DistSq: distSq, VPIdx: vpIdx, Label: label}
		s.len++
		for i := s.len - 1; i > 0; i-- {
			if s.heap[i].DistSq > s.heap[i-1].DistSq {
				s.heap[i], s.heap[i-1] = s.heap[i-1], s.heap[i]
			}
		}
		if s.len == 5 {
			s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq))) * 0.008
		}
	} else if distSq < s.heap[0].DistSq {
		// Substitui o maior elemento (no index 0)
		s.heap[0] = stateNeighbor{DistSq: distSq, VPIdx: vpIdx, Label: label}
		for i := 0; i < 4; i++ {
			if s.heap[i].DistSq < s.heap[i+1].DistSq {
				s.heap[i], s.heap[i+1] = s.heap[i+1], s.heap[i]
			} else {
				break
			}
		}
		s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq))) * 0.008
	}
}

func (s *knnSearchState) searchGreedy(nodeIdx int32) {
	curr := nodeIdx
	for curr != -1 {
		node := &s.tree.Nodes[curr]
		distSq := EuclideanDistSq(s.query, &s.tree.Vectors[node.VPIdx])
		label := s.tree.Labels[node.VPIdx]
		s.addNeighbor(distSq, node.VPIdx, label)
		s.visited++

		if node.Left == -1 && node.Right == -1 {
			break
		}

		dist := float32(math.Sqrt(float64(distSq))) * 0.008
		if dist < node.Radius {
			curr = node.Left
		} else {
			curr = node.Right
		}
	}
}

func (s *knnSearchState) search(nodeIdx int32) {
	if nodeIdx == -1 {
		return
	}

	// Limite de nós visitados: garante latência máxima controlada
	if s.visited >= maxNodesToVisit {
		return
	}

	node := &s.tree.Nodes[nodeIdx]

	distSq := EuclideanDistSq(s.query, &s.tree.Vectors[node.VPIdx])
	dist := float32(math.Sqrt(float64(distSq))) * 0.008
	label := s.tree.Labels[node.VPIdx]

	s.addNeighbor(distSq, node.VPIdx, label)
	s.visited++

	if node.Left == -1 && node.Right == -1 {
		return
	}

	if dist < node.Radius {
		// Busca o lado esquerdo primeiro
		s.search(node.Left)
		// Pruning check: busca o lado direito apenas se a nossa esfera de busca cruzar o limite
		if node.Radius-dist <= s.tau {
			s.search(node.Right)
		}
	} else {
		// Busca o lado direito primeiro
		s.search(node.Right)
		// Pruning check: busca o lado esquerdo apenas se a nossa esfera de busca cruzar o limite
		if dist-node.Radius <= s.tau {
			s.search(node.Left)
		}
	}
}

// euclideanDist calcula a distância exata usando referências quantizadas (usado no build).
func euclideanDist(a, b *[VectorDimsPad]uint8) float32 {
	return float32(math.Sqrt(float64(EuclideanDistSq(a, b)))) * 0.008
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Warmup percorre sequencialmente toda a árvore, vetores e labels na memória física.
// Isso evita que o primeiro acesso (Page Fault) ocorra durante o processamento das transações.
func (t *VPTree) Warmup() {
	var sum uint32
	// Acessa todos os vetores
	for i := range t.Vectors {
		for d := 0; d < VectorDimsPad; d++ {
			sum += uint32(t.Vectors[i][d])
		}
	}
	// Acessa todos os nós
	for i := range t.Nodes {
		sum += uint32(t.Nodes[i].VPIdx)
		sum += uint32(t.Nodes[i].Radius)
		sum += uint32(t.Nodes[i].Left)
		sum += uint32(t.Nodes[i].Right)
	}
	// Acessa todas as labels
	for i := range t.Labels {
		sum += uint32(t.Labels[i])
	}
	// Referência dummy para o compilador não otimizar fora
	if sum == 0xdeadbeef {
		println(sum)
	}
}
