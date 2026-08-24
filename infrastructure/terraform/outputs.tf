output "github_deploy_role_arn" {
  description = "Set this ARN as the demo environment variable AWS_DEPLOY_ROLE_ARN."
  value       = aws_iam_role.github_deploy.arn
}

output "workload_role_arn" {
  description = "Runtime role associated through EKS Pod Identity."
  value       = aws_iam_role.workload.arn
}

output "secret_names" {
  description = "Secret containers whose values must be populated out-of-band."
  value = [
    aws_secretsmanager_secret.database.name,
    aws_secretsmanager_secret.rabbitmq.name,
    aws_secretsmanager_secret.application.name
  ]
}

output "ecr_repository_urls" {
  description = "Private repository URLs; authentication uses IAM, never registry passwords."
  value       = { for name, repository in aws_ecr_repository.component : name => repository.repository_url }
}
