variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "project_name" {
  type    = string
  default = "predictive-maintenance-mlops"
}

variable "cluster_version" {
  type    = string
  default = "1.36"
}

# CIDRs allowed to reach the public EKS API endpoint, usually the operator's
# public IP (x.x.x.x/32). No default on purpose: a personal IP does not belong
# in a public repo. Set it in the git-ignored terraform.tfvars (see
# terraform.tfvars.example); if the IP changes (dynamic ISP, VPN), update it and
# re-apply, or `kubectl` starts timing out.
variable "cluster_public_access_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to reach the public EKS API endpoint. NEVER 0.0.0.0/0."

  validation {
    condition     = length(var.cluster_public_access_cidrs) > 0 && !contains(var.cluster_public_access_cidrs, "0.0.0.0/0")
    error_message = "Set at least one specific CIDR (e.g. your public IP as /32); 0.0.0.0/0 is not allowed."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "db_username" {
  type        = string
  description = "RDS admin user"
  default     = "mlflow_admin"
}

# ------------------------------------------------------------------
# GitLab CI/CD OIDC (see terraform/iam.tf: aws_iam_openid_connect_provider.gitlab)
# ------------------------------------------------------------------
variable "gitlab_oidc_host" {
  type        = string
  description = "GitLab OIDC issuer host (gitlab.com, or your self-managed instance's domain)"
  default     = "gitlab.com"
}

variable "gitlab_project_path" {
  type        = string
  description = "GitLab namespace/project allowed to assume GitLabCIRole (the id_token 'sub' claim). Set it to the real project before applying."
  default     = "personal-group7745334/predictive-maintenance-mlops"
}
