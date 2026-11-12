resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "PostgreSQL from the EKS nodes only"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "PostgreSQL from the EKS nodes (MLflow backend store)"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# ------------------------------------------------------------------
# Terraform-generated RDS credentials, never chosen or committed by hand. The
# value comes from `terraform output` (marked sensitive) and is pasted into the
# Kubernetes Secret (kubernetes/base/secret.yaml; README, "Deploying to AWS").
# It also ends up in the Terraform state, which is why the state bucket is
# CMK-encrypted. A secret manager with automatic rotation is the upgrade once
# the manual step stops being acceptable.
# ------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # safe inside a connection string (postgresql://user:pass@host)
}

resource "aws_db_instance" "mlflow_db" {
  identifier             = "${var.project_name}-backend-db"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  engine                 = "postgres"
  engine_version         = "18.3"
  username               = var.db_username
  password               = random_password.db_password.result
  db_name                = "mlflow_db"
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = var.environment == "dev" ? true : false

  # Single instance, no Multi-AZ or read replicas: this is not a
  # high-availability deployment. Multi-AZ is the upgrade if it has to be.
  multi_az = false

  # Encrypted at rest with the AWS-managed key rather than a CMK: no key policy
  # or rotation of our own to manage.
  storage_encrypted = true

  backup_retention_period = var.environment == "prod" ? 30 : 7
  deletion_protection     = var.environment == "prod" ? true : false
}

output "rds_endpoint" {
  # .address (hostname only), not .endpoint ("host:port"): it feeds DB_HOST, and
  # kubernetes/base/mlflow.yaml appends ":5432" itself; .endpoint would yield an
  # invalid "host:5432:5432".
  value = aws_db_instance.mlflow_db.address
}

output "db_username" {
  value = var.db_username
}

output "db_password" {
  description = "Terraform-generated RDS password, for the Kubernetes Secret (see kubernetes/base/secret.yaml and README.md)."
  value       = random_password.db_password.result
  sensitive   = true
}
