# ============================================================
# EKS node access to S3 (MLflow artifacts, DVC remote)
#
# The node group's instance role, shared by every pod on those nodes, gets the
# bucket permissions through `iam_role_additional_policies` (terraform/eks.tf)
# instead of one IRSA role per ServiceAccount. Less isolated, since any pod on
# the node could use it, but simpler to operate. IRSA per service is the
# upgrade if per-pod isolation becomes a requirement.
# ============================================================
resource "aws_iam_policy" "node_s3_access" {
  name        = "${var.project_name}-node-s3-access"
  description = "EKS node access to the artifacts bucket (MLflow/DVC)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ArtifactsAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}

# ============================================================
# OIDC federation for GitLab CI/CD, with no long-lived static credentials:
# the job's id_token (.gitlab-ci.yml, id_tokens: AWS_OIDC_TOKEN) is exchanged
# through sts:AssumeRoleWithWebIdentity.
# ============================================================
data "tls_certificate" "gitlab" {
  url = "https://${var.gitlab_oidc_host}"
}

resource "aws_iam_openid_connect_provider" "gitlab" {
  url             = "https://${var.gitlab_oidc_host}"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.gitlab.certificates[0].sha1_fingerprint]
}

# ============================================================
# GitLab CI/CD role (GitLabCIRole)
# ============================================================
resource "aws_iam_role" "gitlab_ci_role" {
  name = "GitLabCIRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.gitlab.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.gitlab_oidc_host}:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Only main, the one branch where .gitlab-ci.yml pushes and deploys.
            # A wildcard would let any branch or merge request assume a role
            # with cluster-admin access.
            "${var.gitlab_oidc_host}:sub" = "project_path:${var.gitlab_project_path}:ref_type:branch:ref:main"
          }
        }
      }
    ]
  })
}

# CI policy: ECR push/pull, read/write on the artifacts bucket (DVC), and enough
# EKS access to build a kubeconfig, since the deploy stage runs
# `kubectl apply -k` against the cluster.
resource "aws_iam_policy" "gitlab_ci_policy" {
  name        = "${var.project_name}-gitlab-ci-policy"
  description = "Minimum CI permissions: ECR, S3 and EKS deployment"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "EcrPushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = [aws_ecr_repository.api_repo.arn]
      },
      {
        Sid    = "S3ArtifactsAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        # `aws eks update-kubeconfig` (deploy stage) reads the cluster metadata.
        Sid      = "EksDescribe"
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster"]
        Resource = [module.eks.cluster_arn]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_policy_attachment" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = aws_iam_policy.gitlab_ci_policy.arn
}

# Authorization INSIDE Kubernetes so GitLabCIRole can deploy: AWS
# authentication alone is not enough, EKS also requires an access entry (the
# same mechanism enable_cluster_creator_admin_permissions uses in eks.tf).
# ClusterAdminPolicy is deliberately broad; a namespaced Role/RoleBinding is
# the upgrade for finer-grained permissions.
resource "aws_eks_access_entry" "gitlab_ci" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
}

resource "aws_eks_access_policy_association" "gitlab_ci_admin" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  access_scope {
    type = "cluster"
  }
}

output "gitlab_ci_role_arn" {
  value = aws_iam_role.gitlab_ci_role.arn
}
