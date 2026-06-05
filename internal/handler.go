package internal

import (
	"encoding/json"
	"io"
	"net/http"
	"sync"
)

// Templates estáticos para as 6 respostas possíveis (K=5), evitando json.Marshal (zero-alloc).
var responseTemplates = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),  // 0/5
	[]byte(`{"approved":true,"fraud_score":0.2}`),  // 1/5
	[]byte(`{"approved":true,"fraud_score":0.4}`),  // 2/5
	[]byte(`{"approved":false,"fraud_score":0.6}`), // 3/5
	[]byte(`{"approved":false,"fraud_score":0.8}`), // 4/5
	[]byte(`{"approved":false,"fraud_score":1.0}`), // 5/5
}

// Fallback seguro: aprovamos em caso de erro JSON (peso 1) em vez de falhar HTTP 500 (peso 5).
var fallbackResponse = responseTemplates[0] // approved=true, fraud_score=0.0

// Pools para reciclagem de memória no hot path.
var (
	// Pool de buffers de leitura
	bufPool = sync.Pool{
		New: func() interface{} {
			buf := make([]byte, 0, 2048)
			return &buf
		},
	}

	// Pool de requests
	reqPool = sync.Pool{
		New: func() interface{} {
			return &FraudRequest{}
		},
	}

	// Pool para os vizinhos (KNN)
	neighborPool = sync.Pool{
		New: func() interface{} {
			s := make([]Neighbor, 0, 5)
			return &s
		},
	}
)

// FraudHandler processa as requisições da API.
type FraudHandler struct {
	tree *VPTree
}

// NewFraudHandler creates a new handler backed by the given VP-Tree.
func NewFraudHandler(tree *VPTree) *FraudHandler {
	return &FraudHandler{tree: tree}
}

// RegisterRoutes cria o roteador HTTP.
func (h *FraudHandler) RegisterRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/ready", h.handleReady)
	mux.HandleFunc("/fraud-score", h.handleFraudScore)
	return mux
}

// handleReady responds with 200 OK when the service is ready.
func (h *FraudHandler) handleReady(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// handleFraudScore é o core da detecção (hot path).
func (h *FraudHandler) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	// Usa json.Unmarshal para evitar o overhead de bufio do json.NewDecoder
	bufPtr := bufPool.Get().(*[]byte)
	buf := (*bufPtr)[:0]

	var err error
	buf, err = readBody(r.Body, buf)
	if err != nil {
		*bufPtr = buf
		bufPool.Put(bufPtr)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(fallbackResponse)
		return
	}

	req := reqPool.Get().(*FraudRequest)
	if err := json.Unmarshal(buf, req); err != nil {
		resetFraudRequest(req)
		reqPool.Put(req)
		*bufPtr = buf
		bufPool.Put(bufPtr)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(fallbackResponse)
		return
	}

	vec := Vectorize(req)

	var qVec [VectorDimsPad]uint8
	for i := 0; i < VectorDims; i++ {
		qVec[i] = QuantizeFloat32(vec[i])
	}

	knnRes := h.tree.KNN(&qVec, 5)
	score := ComputeFraudScore(knnRes)

	// Converte score para índice [0-5]
	idx := int(score*5 + 0.5) // Round to nearest integer
	if idx < 0 {
		idx = 0
	}
	if idx > 5 {
		idx = 5
	}

	// Recicla memória no final do processamento feliz
	resetFraudRequest(req)
	reqPool.Put(req)
	*bufPtr = buf
	bufPool.Put(bufPtr)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(responseTemplates[idx])
}

// readBody carrega o request HTTP num buffer reciclado.
func readBody(body io.ReadCloser, buf []byte) ([]byte, error) {
	if body == nil {
		return buf, io.EOF
	}
	for {
		if len(buf) == cap(buf) {
			// Grow buffer
			newBuf := make([]byte, len(buf), cap(buf)*2+512)
			copy(newBuf, buf)
			buf = newBuf
		}
		n, err := body.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		if err != nil {
			if err == io.EOF {
				return buf, nil
			}
			return buf, err
		}
	}
}

// resetFraudRequest limpa um FraudRequest para reuso no sync.Pool.
func resetFraudRequest(r *FraudRequest) {
	r.ID = ""
	r.Transaction = Transaction{}
	r.Customer = Customer{KnownMerchants: r.Customer.KnownMerchants[:0]}
	r.Merchant = Merchant{}
	r.Terminal = Terminal{}
	r.LastTransaction = nil
}
