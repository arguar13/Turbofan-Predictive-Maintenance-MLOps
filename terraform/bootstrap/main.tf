# Bootstrap of the remote Terraform backend: the versioned, encrypted S3 bucket
# and the DynamoDB lock table that ../provider.tf's "s3" backend uses. A remote
# backend cannot create its own bucket, so this is applied ONCE, with purely
# local state, before the rest of the project.
#
# Usage (once per account/environment):
#   cd terraform/bootstrap
#   terraform init
#   terraform apply
#   terraform output -raw backend_config > ../backend.hcl
#   # then, in terraform/ (backend.hcl is git-ignored):
#   terraform init -backend-config=backend.hcl -migrate-state

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "predictive-maintenance-mlops"
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "tfstate" {
  bucket = "${var.project_name}-tfstate-${data.aws_caller_identity.current.account_id}"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      # AWS-0132 (HIGH): our own CMK (kms.tf) instead of the AWS-managed key.
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.tfstate.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tfstate_lock" {
  name         = "${var.project_name}-tfstate-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  # AWS-0025: encrypted at rest with the backend's CMK.
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.tfstate.arn
  }

  # AWS-0024: point-in-time recovery.
  point_in_time_recovery {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

output "backend_config" {
  value = <<-EOT
    bucket         = "${aws_s3_bucket.tfstate.bucket}"
    key            = "${var.project_name}/terraform.tfstate"
    region         = "${var.aws_region}"
    dynamodb_table = "${aws_dynamodb_table.tfstate_lock.name}"
    encrypt        = true
  EOT
}
