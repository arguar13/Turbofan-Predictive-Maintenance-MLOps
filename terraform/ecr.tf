# The project's only ECR repository: MLflow runs the public ghcr.io image
# (kubernetes/base/mlflow.yaml, docker-compose.yml), and CI uses a plain
# `docker build` + `docker push` with no remote BuildKit cache.
resource "aws_ecr_repository" "api_repo" {
  name = "${var.project_name}-api"
  # AWS-0031 (HIGH): immutable tags. With mutable ones, anyone allowed to push
  # could silently replace a deployed tag, including the commit SHA pinned in
  # the Kustomize overlay. Re-pushing a SHA now fails loudly instead.
  image_tag_mutability = "IMMUTABLE"
  # The stack goes through frequent destroy/apply cycles; without force_delete,
  # `terraform destroy` fails while the repository still holds images.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }

  # Encrypted with ECR's default AWS-managed AES256 key.
}

resource "aws_ecr_lifecycle_policy" "repo_cleanup" {
  repository = aws_ecr_repository.api_repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

output "ecr_api_repository_url" {
  value = aws_ecr_repository.api_repo.repository_url
}
