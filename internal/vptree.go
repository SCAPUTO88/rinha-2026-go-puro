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

type KNNResult struct {
	Neighbors [5]Neighbor
	Len       int
}

// KNN encontra os k-vizinhos mais próximos aplicando poda por desigualdade triangular de forma zero-alloc.
func (t *VPTree) KNN(query *[VectorDimsPad]float32, k int) KNNResult {
	if t.Root == -1 {
		return KNNResult{}
	}

	s := knnStatePool.Get().(*knnSearchState)
	s.tree = t
	s.query = query
	s.k = k
	s.len = 0
	s.tau = float32(math.Inf(1))
	s.heap = [5]Neighbor{} // zera heap anterior

	s.search(t.Root)

	// Copia o resultado
	var result KNNResult
	result.Neighbors = s.heap
	result.Len = s.len

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
	tree  *VPTree
	query *[VectorDimsPad]float32
	k     int
	heap  [5]Neighbor // Array fixo
	len   int
	tau   float32     // Worst actual distance (not squared)
}

func (s *knnSearchState) addNeighbor(distSq float32, label uint8) {
	if s.len < 5 {
		// Insere mantendo ordenado decrescentemente por DistSq
		s.heap[s.len] = Neighbor{DistSq: distSq, Label: label}
		s.len++
		for i := s.len - 1; i > 0; i-- {
			if s.heap[i].DistSq > s.heap[i-1].DistSq {
				s.heap[i], s.heap[i-1] = s.heap[i-1], s.heap[i]
			}
		}
		if s.len == 5 {
			s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq)))
		}
	} else if distSq < s.heap[0].DistSq {
		// Substitui o maior elemento (no index 0)
		s.heap[0] = Neighbor{DistSq: distSq, Label: label}
		for i := 0; i < 4; i++ {
			if s.heap[i].DistSq < s.heap[i+1].DistSq {
				s.heap[i], s.heap[i+1] = s.heap[i+1], s.heap[i]
			} else {
				break
			}
		}
		s.tau = float32(math.Sqrt(float64(s.heap[0].DistSq)))
	}
}

func (s *knnSearchState) search(nodeIdx int32) {
	if nodeIdx == -1 {
		return
	}

	node := &s.tree.Nodes[nodeIdx]

	// Compute actual distance from query to vantage point
	distSq := EuclideanDistSq(s.query, &s.tree.Vectors[node.VPIdx])
	dist := float32(math.Sqrt(float64(distSq)))
	label := s.tree.Labels[node.VPIdx]

	// Consider this vantage point as a candidate neighbor
	s.addNeighbor(distSq, label)

	// No children — nothing more to search
	if node.Left == -1 && node.Right == -1 {
		return
	}

	// Determine search order and pruning
	if dist < node.Radius {
		// Query is inside the ball — search Left (inside) first
		s.search(node.Left)
		// Search Right (outside) only if our search sphere crosses the ball boundary
		if dist+s.tau >= node.Radius {
			s.search(node.Right)
		}
	} else {
		// Query is outside the ball — search Right (outside) first
		s.search(node.Right)
		// Search Left (inside) only if our search sphere crosses the ball boundary
		if dist-s.tau <= node.Radius {
			s.search(node.Left)
		}
	}
}

// euclideanDist calcula a distância exata usando referências quantizadas (usado no build).
func euclideanDist(a, b *[VectorDimsPad]uint8) float32 {
	return float32(math.Sqrt(float64(EuclideanDistSqRefRef(a, b))))
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
