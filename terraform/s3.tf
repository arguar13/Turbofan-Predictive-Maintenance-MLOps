resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
  # Same reasoning as force_delete in terraform/ecr.tf: the stack goes through
  # full destroy/apply cycles, and `terraform destroy` would otherwise fail
  # while the bucket holds objects or old versions (versioning is enabled).
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts_encryption" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      # SSE-S3 (AWS-managed AES256) instead of a CMK: still encrypted at rest;
      # what is lost is our own rotation, key policy and per-key CloudTrail
      # attribution.
      sse_algorithm = "AES256"
    }
  }
}

# Model artifacts and DVC data must never be public (trivy AWS-0094).
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Account ID, used to make the bucket name globally unique
data "aws_caller_identity" "current" {}

output "s3_bucket_name" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}
