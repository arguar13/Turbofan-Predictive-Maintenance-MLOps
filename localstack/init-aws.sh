#!/usr/bin/env bash
# LocalStack bootstrap: runs once when LocalStack is ready (native
# /etc/localstack/init/ready.d/*.sh hook) and creates the S3 bucket that
# terraform/s3.tf creates in AWS, so the stack runs against a local S3 with no
# real resources or credentials.
set -euo pipefail

BUCKET_NAME="${MLOPS_S3_BUCKET:-predictive-maintenance-mlops-artifacts-local}"

echo "[localstack-init] Creating S3 bucket: ${BUCKET_NAME}"
awslocal s3 mb "s3://${BUCKET_NAME}"
awslocal s3api put-bucket-versioning \
  --bucket "${BUCKET_NAME}" \
  --versioning-configuration Status=Enabled

echo "[localstack-init] Done: bucket=${BUCKET_NAME}"
