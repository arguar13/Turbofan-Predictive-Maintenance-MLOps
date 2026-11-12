terraform {
  required_version = ">= 1.5.0"

  # Remote backend created by terraform/bootstrap/ (versioned S3 bucket
  # encrypted with a CMK, plus a DynamoDB lock table).
  #
  # Partial configuration: the block is empty on purpose, because the bucket
  # name contains the account ID, which has no place in a public repo. The
  # values come from the bootstrap's "backend_config" output into a local,
  # git-ignored backend.hcl (see backend.hcl.example):
  #
  #   terraform -chdir=bootstrap output -raw backend_config > backend.hcl
  #   terraform init -backend-config=backend.hcl
  #
  # State no longer lives on a laptop, and the DynamoDB lock stops two
  # concurrent applies from corrupting it.
  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "Terraform"
    }
  }
}
