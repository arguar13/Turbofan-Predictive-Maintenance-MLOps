module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  # No "-cluster" suffix: the module appends its own "-cluster-" to the IAM
  # role's name_prefix, which is limited to 38 characters, and
  # "${var.project_name}-cluster" plus that suffix exceeds it.
  cluster_name    = var.project_name
  cluster_version = var.cluster_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  # Kubernetes API access.
  #
  # AWS-0041 (CRITICAL): by default the module opens the public endpoint to
  # 0.0.0.0/0. It is restricted to var.cluster_public_access_cidrs.
  #
  # The endpoint stays public on purpose (AWS-0040): a private one would need a
  # bastion or VPN for every `kubectl`, including the CI deploy stage.
  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = var.cluster_public_access_cidrs

  # AWS-0038: without control-plane logs in CloudWatch there is no trail of
  # authentication, authorization or API calls to investigate an incident.
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Pods reach S3 through the node group's instance role (ml_workers below and
  # terraform/iam.tf: aws_iam_policy.node_s3_access), not one IRSA role per
  # ServiceAccount: simpler to operate, at the cost of isolating the permission
  # per node instead of per pod. IRSA is the upgrade if per-pod isolation
  # becomes a requirement.
  enable_irsa = false

  # Module v20 no longer makes the identity running `terraform apply` a cluster
  # admin: without this, `kubectl get nodes` answers "the server has asked for
  # the client to provide credentials" even though the cluster is ACTIVE. That
  # is authorization inside Kubernetes (access entries), not AWS
  # authentication. This creates the access entry for whoever applies Terraform.
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    ml_workers = {
      min_size       = 1
      max_size       = 3
      desired_size   = 2
      instance_types = ["t3.large"]
      disk_size      = 40

      # See enable_irsa above: mlflow and api reach S3 through this role.
      iam_role_additional_policies = {
        s3_access = aws_iam_policy.node_s3_access.arn
      }
    }
  }
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}
