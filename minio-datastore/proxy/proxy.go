package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

type Scenario struct {
	Name          string        `json:"name"`
	Dataset       string        `json:"dataset"`
	Preprocessing Preprocessing `json:"preprocessing"`
	Model         []Layer       `json:"model"`
}

type Preprocessing struct {
	ExcludeColumns []string `json:"exclude_columns"`
	TrainTestSplit float64  `json:"train_test_split"`
	Scaling        string   `json:"scaling"`
	Normalize      bool     `json:"normalize"`
}

type Layer struct {
	ClassName string                 `json:"class_name"`
	Config    map[string]interface{} `json:"config"`
}

var ErrDatasetNotFound = errors.New("dataset could not be found in Kaggle")

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

func fileExistsInMinio(client *minio.Client, filename string) bool {
	_, err := client.StatObject(context.Background(), MinioBucket, filename, minio.StatObjectOptions{})

	if err == nil {
		log.Printf("File %s found in Minio", filename)
	} else {
		log.Printf("File %s not found in Minio", filename)
	}

	return err == nil
}

func extractParquetFromZip(zipname string, outName string) error {
	archive, err := zip.OpenReader(zipname)
	if err != nil {
		log.Printf("Could not open zip: %v", err)
		return err
	}
	defer archive.Close()

	log.Println("Extracting parquet file from zip")
	for _, f := range archive.File {
		if strings.HasSuffix(f.Name, ".parquet") {
			fileInArchive, err := f.Open()
			if err != nil {
				return err
			}

			dstFile, err := os.Create(outName)
			if err != nil {
				return err
			}

			if _, err := io.Copy(dstFile, fileInArchive); err != nil {
				return err
			}

			dstFile.Close()
			fileInArchive.Close()
		}
	}

	removeLocal(zipname)

	return nil
}

func scenarioHandler(w http.ResponseWriter, r *http.Request, client *minio.Client) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	if r.Header.Get("Content-Type") != "application/json" {
		http.Error(w, "Expected Content-Type application/json", http.StatusUnsupportedMediaType)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Could not read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var scenario Scenario
	if err := json.Unmarshal(body, &scenario); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	// Create scenarioFilename
	scenarioFilename := scenario.Name + ".json"

	// Write original JSON to file
	if err := os.WriteFile(scenarioFilename, body, 0644); err != nil {
		http.Error(w, "Failed to save scenario", http.StatusInternalServerError)
		return
	}

	log.Printf("Scenario saved locally as %s\n", scenarioFilename)

	datasetFilename := scenario.Dataset + ".parquet"
	minioDatasetFilename := scenario.Dataset + "/" + datasetFilename
	minioScenarioFilename := scenario.Dataset + "/" + scenarioFilename

	// Check if dataset already present
	if !fileExistsInMinio(client, minioDatasetFilename) {
		zipName, err := downloadFromKaggle(scenario.Dataset)
		if err != nil {
			if err == ErrDatasetNotFound {
				http.Error(w, "Dataset does not exist on Kaggle", http.StatusNotFound)
			} else {
				http.Error(w, "Failed to download from Kaggle", http.StatusInternalServerError)
			}
			return
		}

		if err := extractParquetFromZip(zipName, datasetFilename); err != nil {
			http.Error(w, "Failed to unzip", http.StatusInternalServerError)
			return
		}

		if err := uploadToMinio(client, datasetFilename, minioDatasetFilename); err != nil {
			http.Error(w, "Failed to upload dataset to Minio", http.StatusInternalServerError)
			return
		} else {
			removeLocal(datasetFilename)
		}

		if err := uploadToMinio(client, scenarioFilename, minioScenarioFilename); err != nil {
			http.Error(w, "Failed to upload scenario to Minio", http.StatusInternalServerError)
			return
		}
	}

	removeLocal(scenarioFilename)

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(fmt.Sprintf("Scenario saved as %s", scenarioFilename)))
}

func downloadFromKaggle(kaggleDatasetName string) (string, error) {
	// fileKey ex "dhoogla-nftoniotv2"
	fileKeyURL := KaggleURL + strings.Replace(kaggleDatasetName, "-", "/", -1)

	log.Printf("Downloading %s from Kaggle", kaggleDatasetName)
	resp, err := http.Get(fileKeyURL)
	if err != nil {
		log.Printf("Failed to download from Kaggle: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusNotFound {
		log.Printf("Dataset not found on Kaggle: %s", kaggleDatasetName)
		return "", ErrDatasetNotFound
	}

	//TODO: use WriteFile
	outName := kaggleDatasetName + ".zip"
	file, err := os.Create(outName)
	if err != nil {
		log.Printf("Failed to create file: %v", err)
		return "", err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		log.Printf("Failed to save file: %v", err)
	}

	return outName, err
}

func uploadToMinio(client *minio.Client, srcFilename string, dstFilename string) error {
	log.Printf("Uploading %s to Minio", srcFilename)

	_, err := client.FPutObject(context.Background(), MinioBucket, dstFilename, srcFilename, minio.PutObjectOptions{})
	if err != nil {
		log.Printf("Failed to upload to Minio: %v", err)
	}

	return err
}

func removeLocal(filename string) error {
	log.Printf("Removing file %s", filename)
	err := os.Remove(filename)
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

func main() {
	client, err := minio.New(MinioEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(MinioAccessKey, MinioSecretKey, ""),
		Secure: false,
	})
	if err != nil {
		log.Fatalf("Failed to create Minio client: %v", err)
	}

	ensureBucketExists(client)

	http.HandleFunc("/scenario", func(w http.ResponseWriter, r *http.Request) {
		scenarioHandler(w, r, client)
	})

	log.Println("Server started on ", ListenPort)
	log.Fatal(http.ListenAndServe(ListenPort, nil))
}
