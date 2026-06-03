//go:build !windows

package internal

import (
	"os"
	"syscall"
)

// mmapFile mapeia o arquivo em memória para compartilhamento entre processos (Linux).
func mmapFile(path string) ([]byte, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}

	size := int(fi.Size())
	if size == 0 {
		return nil, func() {}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}

	cleanup := func() {
		syscall.Munmap(data)
	}

	return data, cleanup, nil
}
