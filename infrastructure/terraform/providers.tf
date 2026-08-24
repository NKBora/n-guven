provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com"
}
