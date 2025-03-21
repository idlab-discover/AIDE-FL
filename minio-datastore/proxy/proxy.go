package main

import (
	"context"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

const (
	ListenPort     = ":8080"
	MinioEndpoint  = "localhost:9000"
	MinioAccessKey = "aide-fl1"
	MinioSecretKey = "aide-fl1"
	MinioBucket    = "datasets"
	KaggleURL      = "https://www.kaggle.com/api/v1/datasets/download/" // Modify for dataset access
)

var ErrDatasetNotFound = errors.New("dataset could not be found in Kaggle")

func main() {
	client, err := minio.New(MinioEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(MinioAccessKey, MinioSecretKey, ""),
		Secure: false,
	})
	if err != nil {
		log.Fatalf("Failed to create Minio client: %v", err)
	}

	ensureBucketExists(client)

	http.HandleFunc("/download", func(w http.ResponseWriter, r *http.Request) {
		fileKey := r.URL.Query().Get("file")
		if fileKey == "" {
			http.Error(w, "Missing file parameter", http.StatusBadRequest)
			return
		}

		if !fileExistsInMinio(client, fileKey) {
			if err := downloadFromKaggle(fileKey); err != nil {
				if err == ErrDatasetNotFound {
					http.Error(w, "Dataset does not exist on Kaggle", http.StatusNotFound)
				} else {
					http.Error(w, "Failed to download from Kaggle", http.StatusInternalServerError)
				}
				return
			}
			if err := uploadToMinio(client, fileKey); err != nil {
				http.Error(w, "Failed to upload to Minio", http.StatusInternalServerError)
				return
			}
		}

		serveFromMinio(client, w, fileKey)
	})

	log.Println("Server started on ", ListenPort)
	log.Fatal(http.ListenAndServe(ListenPort, nil))
}

func ensureBucketExists(client *minio.Client) {
	exists, err := client.BucketExists(context.Background(), MinioBucket)
	if err != nil {
		log.Fatalf("Error checking bucket existence: %v", err)
	}
	if !exists {
		log.Printf("Bucket %s does not exist, creating...", MinioBucket)
		err = client.MakeBucket(context.Background(), MinioBucket, minio.MakeBucketOptions{})
		if err != nil {
			log.Fatalf("Failed to create bucket: %v", err)
		}
	} else {
		log.Printf("Bucket %s exists", MinioBucket)
	}
}

func fileExistsInMinio(client *minio.Client, fileKey string) bool {
	_, err := client.StatObject(context.Background(), MinioBucket, fileKey+".zip", minio.StatObjectOptions{})

	if err == nil {
		log.Printf("File %s.zip found in Minio", fileKey)
	} else {
		log.Printf("File %s.zip not found in Minio", fileKey)
	}

	return err == nil
}

func downloadFromKaggle(fileKey string) error {
	// fileKey ex "dhoogla-nftoniotv2"
	fileKeyURL := KaggleURL + strings.Replace(fileKey, "-", "/", -1)

	log.Printf("Downloading %s from Kaggle", fileKey)
	resp, err := http.Get(fileKeyURL)
	if err != nil {
		log.Printf("Failed to download from Kaggle: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusNotFound {
		log.Printf("Dataset not found on Kaggle: %s", fileKey)
		return ErrDatasetNotFound
	}

	file, err := os.Create(fileKey)
	if err != nil {
		log.Printf("Failed to create file: %v", err)
		return err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		log.Printf("Failed to save file: %v", err)
	}

	return err
}

func uploadToMinio(client *minio.Client, fileKey string) error {
	log.Printf("Uploading %s to Minio", fileKey)

	_, err := client.FPutObject(context.Background(), MinioBucket, fileKey+".zip", fileKey, minio.PutObjectOptions{})
	if err != nil {
		log.Printf("Failed to upload to Minio: %v", err)
	}
	defer removeLocal(fileKey)

	return err
}

func removeLocal(fileKey string) error {
	err := os.Remove(fileKey)
	if err != nil {
		log.Printf("Failed to remove file: %v", err)
	}
	return err
}

func serveFromMinio(client *minio.Client, w http.ResponseWriter, fileKey string) {
	log.Printf("Serving %s from Minio", fileKey)

	obj, err := client.GetObject(context.Background(), MinioBucket, fileKey+".zip", minio.GetObjectOptions{})
	if err != nil {
		http.Error(w, "Failed to fetch file from Minio", http.StatusInternalServerError)
		return
	}
	defer obj.Close()

	w.Header().Set("Content-Disposition", "attachment; filename="+fileKey+".zip")
	w.WriteHeader(http.StatusOK)
	io.Copy(w, obj)
}
