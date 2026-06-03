//go:build windows

package internal

import "os"

// mmapFile no Windows faz fallback usando alocação regular em RAM.
func mmapFile(path string) ([]byte, func(), error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return data, func() {}, nil
}
