variable "aws_region" {
  description = "AWS region for the demo resources."
  type        = string
  default     = "eu-west-1"
}

variable "github_repository" {
  description = "GitHub repository in owner/name form; used to restrict OIDC trust."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository must use owner/name format."
  }
}

variable "github_owner_id" {
  description = "Immutable numeric GitHub owner ID used by repositories created after 15 July 2026."
  type        = string

  validation {
    condition     = can(regex("^[0-9]+$", var.github_owner_id))
    error_message = "github_owner_id must be a numeric string."
  }
}

variable "github_repository_id" {
  description = "Immutable numeric GitHub repository ID used in the OIDC subject claim."
  type        = string

  validation {
    condition     = can(regex("^[0-9]+$", var.github_repository_id))
    error_message = "github_repository_id must be a numeric string."
  }
}

variable "github_environment" {
  description = "GitHub Environment allowed to assume the deployment role."
  type        = string
  default     = "demo"
}

variable "create_github_oidc_provider" {
  description = "Create the account-wide GitHub OIDC provider. Set false when it already exists."
  type        = bool
  default     = true
}

variable "existing_github_oidc_provider_arn" {
  description = "Existing GitHub OIDC provider ARN when create_github_oidc_provider is false."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition     = var.create_github_oidc_provider || var.existing_github_oidc_provider_arn != null
    error_message = "Provide existing_github_oidc_provider_arn when provider creation is disabled."
  }
}

variable "eks_cluster_name" {
  description = "Existing EKS cluster that receives the demo baseline."
  type        = string
}

variable "kubernetes_namespace" {
  description = "Namespace used by the N-Güven demo workloads."
  type        = string
  default     = "nguven-demo"
}

variable "kubernetes_service_account" {
  description = "Service account associated with the runtime IAM role."
  type        = string
  default     = "nguven-workload"
}

variable "ecr_repository_names" {
  description = "Private ECR repositories prepared for independently deployable components."
  type        = set(string)
  default = [
    "nguven/backend",
    "nguven/web",
    "nguven/text-ai",
    "nguven/image-ai",
    "nguven/public-figure-ai"
  ]
}

variable "secret_prefix" {
  description = "Secrets Manager path prefix. Secret values are never managed by this module."
  type        = string
  default     = "nguven/demo"
}

variable "tags" {
  description = "Tags applied to managed AWS resources."
  type        = map(string)
  default = {
    Project     = "N-Guven"
    Environment = "demo"
    ManagedBy   = "Terraform"
  }
}
