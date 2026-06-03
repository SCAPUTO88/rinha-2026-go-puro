package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"rinha-2026/internal"
	"time"
)

func main() {
	// Health check mode usado pelo Docker
	if len(os.Args) > 1 && os.Args[1] == "--health" {
		port := os.Getenv("PORT")
		if port == "" {
			port = "8080"
		}
		conn, err := net.DialTimeout("tcp", "127.0.0.1:"+port, time.Second)
		if err != nil {
			os.Exit(1)
		}
		conn.Close()
		os.Exit(0)
	}

	// Resolve o caminho do binário (args ou env)
	binPath := os.Getenv("VPTREE_BIN")
	if binPath == "" && len(os.Args) > 1 {
		binPath = os.Args[1]
	}
	if binPath == "" {
		binPath = "/data/vptree.bin" // Padrão no Docker
	}

	// Define a porta do servidor
	port := os.Getenv("PORT")
	if port == "" {
		port = "9999"
	}

	log.Printf("loading VP-Tree from %s...", binPath)
	tree, cleanup, err := internal.LoadVPTreeBinary(binPath)
	if err != nil {
		log.Fatalf("failed to load VP-Tree: %v", err)
	}
	defer cleanup()

	log.Printf("VP-Tree loaded: %d nodes, %d vectors", len(tree.Nodes), len(tree.Vectors))

	handler := internal.NewFraudHandler(tree)
	mux := handler.RegisterRoutes()

	addr := fmt.Sprintf(":%s", port)
	log.Printf("listening on %s", addr)

	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
