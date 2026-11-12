# ============================================================
# Deployment identity (IaC), so the root account is no longer used
#
# AWS states that the root account must not have access keys or be used for
# operational work: it cannot be scoped by policies, cannot be revoked without
# changing the account password, and all its CloudTrail activity shows up as
# "root" with no useful attribution.
#
# This bootstrap is the ONLY part of the project applied with root
# credentials, and its purpose is to end that need: it creates an IAM user with
# its own access key, used by `terraform/` (the real stack) and the CLI from
# then on.
#
# Group + user rather than an inline user policy, so adding a second operator
# is one line, not a copy of the policy.
# ============================================================

resource "aws_iam_group" "deployers" {
  name = "${var.project_name}-deployers"
  path = "/mlops/"
}

# AdministratorAccess: terraform/ provisions the VPC, EKS, RDS, S3, ECR and
# CloudWatch Logs and creates IAM roles, policies and an OIDC provider. A
# hand-scoped policy would be huge, fragile and need editing with every change.
# The real blast-radius limit is that this identity CAN be rotated, audited by
# name in CloudTrail and revoked in seconds, none of which holds for root.
resource "aws_iam_group_policy_attachment" "deployers_admin" {
  group      = aws_iam_group.deployers.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_user" "deployer" {
  name = "${var.project_name}-deployer"
  path = "/mlops/"

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
    # No ";": IAM tag values only allow [\p{L}\p{Z}\p{N}_.:/=+\-@], and a
    # semicolon makes CreateUser fail with ValidationError.
    Purpose = "IaC deployment identity. Replaces use of the root account"
  }
}

resource "aws_iam_user_group_membership" "deployer" {
  user   = aws_iam_user.deployer.name
  groups = [aws_iam_group.deployers.name]
}

resource "aws_iam_access_key" "deployer" {
  user = aws_iam_user.deployer.name
}

# ------------------------------------------------------------------
# Outputs
#
# SECURITY NOTE: the secret lives in THIS directory's Terraform state, which is
# local and git-ignored (*.tfstate): the unavoidable price of creating a static
# credential through IaC. Rotation:
# `terraform taint aws_iam_access_key.deployer && terraform apply`.
# ------------------------------------------------------------------
output "deployer_user_name" {
  description = "IAM user to use instead of root for terraform/ and the AWS CLI"
  value       = aws_iam_user.deployer.name
}

output "deployer_access_key_id" {
  description = "AWS_ACCESS_KEY_ID of the deployment user"
  value       = aws_iam_access_key.deployer.id
}

output "deployer_secret_access_key" {
  description = "AWS_SECRET_ACCESS_KEY of the deployment user"
  value       = aws_iam_access_key.deployer.secret
  sensitive   = true
}
