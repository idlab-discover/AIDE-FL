#!/bin/bash

set -e

CERT_DIR=certs

cd "$(dirname "$0")"
mkdir -p $CERT_DIR
cd $CERT_DIR

# Get server's IP to include in cert
SERVER_IP=$(kubectl get nodes -l fl-role=server -o jsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}')

# Private key for CA
openssl ecparam -genkey -name secp384r1 -out ca.key

# Self-signed CA certificate
openssl req -x509 -new -key ca.key -sha384 -days 365 -out ca.crt -subj "/"

# Private key for server
openssl ecparam -genkey -name secp384r1 -out server.key

# CSR
openssl req -new -key server.key -sha384 -out server.csr -subj "/" -addext "subjectAltName=DNS.1:localhost,IP.1:::1,IP.2:127.0.0.1,IP.3:0.0.0.0,IP.4:$SERVER_IP"

# Sign CSR
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.pem -days 365 -sha384 -copy_extensions copyall