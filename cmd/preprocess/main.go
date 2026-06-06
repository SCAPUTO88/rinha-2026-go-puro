package main

import (
	"fmt"
	"log"
	"os"
	"rinha-2026/internal"
	"time"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: preprocess <input.json.gz> <output.bin>\n")
		os.Exit(1)
	}

	inputPath := os.Args[1]
	outputPath := os.Args[2]

	log.Printf("loading references from %s...", inputPath)
	start := time.Now()

	ds, err := internal.LoadReferencesAsBFDataset(inputPath)
	if err != nil {
		log.Fatalf("failed to load references: %v", err)
	}
	log.Printf("loaded %d references in %v", ds.NumRefs, time.Since(start))

	log.Printf("serializing to %s...", outputPath)
	start = time.Now()
	if err := internal.SerializeBFDataset(ds, outputPath); err != nil {
		log.Fatalf("failed to serialize: %v", err)
	}

	fi, _ := os.Stat(outputPath)
	log.Printf("done! wrote %d bytes (%.1f MB) in %v",
		fi.Size(), float64(fi.Size())/1024/1024, time.Since(start))
}
