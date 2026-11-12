# ============================================================
# CMK for the Terraform state backend
#
# The state holds endpoints, ARNs and the generated RDS password
# (random_password) in plain text. SSE-S3 would leave trivy's AWS-0132 (HIGH)
# open and, more importantly, give no control over who can decrypt it and no
# rotation of our own.
#
# The main stack (terraform/) uses AWS-managed keys only; this key exists for
# the state backend and lives here because the bootstrap must apply to an empty
# account, before any other resource exists.
# ============================================================

resource "aws_kms_key" "tfstate" {
  description             = "CMK for the ${var.project_name} Terraform state backend"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "tfstate" {
  name          = "alias/${var.project_name}-tfstate"
  target_key_id = aws_kms_key.tfstate.key_id
}
