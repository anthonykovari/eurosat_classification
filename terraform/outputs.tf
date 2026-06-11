output "data_lake_bucket" {
  description = "S3 bucket for the EuroSAT data lake"
  value       = aws_s3_bucket.data_lake.bucket
}

output "model_registry_bucket" {
  description = "S3 bucket for the model registry"
  value       = aws_s3_bucket.model_registry.bucket
}

output "ecr_backend_url" {
  description = "ECR repository URL for the backend image"
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_frontend_url" {
  description = "ECR repository URL for the frontend image"
  value       = aws_ecr_repository.frontend.repository_url
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API server endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "sagemaker_role_arn" {
  description = "IAM role ARN for SageMaker execution"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}
