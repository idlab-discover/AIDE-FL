package custom

import (
	"bytes"
	"fmt"
	"sync"

	"github.com/xitongsys/parquet-go/source"
)

type InMemoryFile struct {
	buf *bytes.Reader
	mu  sync.Mutex
}

func NewInMemoryFile(data []byte) *InMemoryFile {
	return &InMemoryFile{buf: bytes.NewReader(data)}
}

func (f *InMemoryFile) Seek(offset int64, whence int) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.buf.Seek(offset, whence)
}

func (f *InMemoryFile) Read(p []byte) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.buf.Read(p)
}

func (f *InMemoryFile) Close() error {
	// Nothing to do for in-memory
	return nil
}

func (f *InMemoryFile) Write(p []byte) (int, error) {
	return 0, fmt.Errorf("not writable")
}

func (f *InMemoryFile) Open(name string) (source.ParquetFile, error) {
	return f, nil
}

func (f *InMemoryFile) Create(name string) (source.ParquetFile, error) {
	return nil, fmt.Errorf("create not supported")
}

func (f *InMemoryFile) Name() string {
	return "in-memory-file"
}
