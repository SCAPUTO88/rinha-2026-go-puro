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
		socketPath := os.Getenv("SOCKET_PATH")
		if socketPath != "" {
			conn, err := net.DialTimeout("unix", socketPath, time.Second)
			if err != nil {
				os.Exit(1)
			}
			conn.Close()
			os.Exit(0)
		}
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
	binPath := os.Getenv("REFS_BIN")
	if binPath == "" && len(os.Args) > 1 {
		binPath = os.Args[1]
	}
	if binPath == "" {
		binPath = "/data/refs.bin" // Padrão no Docker
	}

	log.Printf("loading BFDataset from %s...", binPath)
	ds, cleanup, err := internal.LoadBFDataset(binPath)
	if err != nil {
		log.Fatalf("failed to load BFDataset: %v", err)
	}
	defer cleanup()

	log.Printf("warming up BFDataset...")
	startWarmup := time.Now()
	ds.Warmup()
	log.Printf("BFDataset warmed up in %v", time.Since(startWarmup))

	log.Printf("BFDataset loaded: %d references, %d dims", ds.NumRefs, internal.BFDims)

	handler := internal.NewFraudHandler(ds)
	mux := handler.RegisterRoutes()

	server := &http.Server{
		Handler: mux,
	}

	// Check for Unix socket mode
	socketPath := os.Getenv("SOCKET_PATH")
	if socketPath != "" {
		// Remove stale socket file if it exists
		os.Remove(socketPath)
		ln, err := net.Listen("unix", socketPath)
		if err != nil {
			log.Fatalf("failed to listen on unix socket %s: %v", socketPath, err)
		}
		// Make socket world-readable/writable for nginx
		os.Chmod(socketPath, 0777)
		log.Printf("listening on unix socket %s", socketPath)
		if err := server.Serve(ln); err != nil {
			log.Fatalf("server error: %v", err)
		}
	} else {
		// TCP mode
		port := os.Getenv("PORT")
		if port == "" {
			port = "9999"
		}
		addr := fmt.Sprintf(":%s", port)
		server.Addr = addr
		log.Printf("listening on %s", addr)
		if err := server.ListenAndServe(); err != nil {
			log.Fatalf("server error: %v", err)
		}
	}
}
