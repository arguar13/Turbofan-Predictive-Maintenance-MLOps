module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs              = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24"]

  create_database_subnet_group = true

  # Egress for the private EKS nodes (images, packages). One NAT gateway instead
  # of one per AZ: cheaper, but private egress is lost if that AZ goes down.
  enable_nat_gateway = true
  single_nat_gateway = true

  enable_dns_hostnames = true
  enable_dns_support   = true
}
