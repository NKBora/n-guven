locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : var.existing_github_oidc_provider_arn
  github_repository_parts  = split("/", var.github_repository)
  github_subject           = "repo:${local.github_repository_parts[0]}@${var.github_owner_id}/${local.github_repository_parts[1]}@${var.github_repository_id}:environment:${var.github_environment}"
}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[0].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_deploy_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [local.github_subject]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "nguven-demo-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_deploy_trust.json
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "EcrAuthorization"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PushNGuvenImages"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart"
    ]
    resources = [for repository in var.ecr_repository_names : "arn:${data.aws_partition.current.partition}:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${repository}"]
  }

  statement {
    sid       = "DescribeTargetCluster"
    effect    = "Allow"
    actions   = ["eks:DescribeCluster"]
    resources = ["arn:${data.aws_partition.current.partition}:eks:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster/${var.eks_cluster_name}"]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "nguven-demo-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}

resource "aws_eks_access_entry" "github_deploy" {
  cluster_name  = var.eks_cluster_name
  principal_arn = aws_iam_role.github_deploy.arn
  type          = "STANDARD"
}

resource "aws_eks_access_policy_association" "github_deploy" {
  cluster_name  = var.eks_cluster_name
  principal_arn = aws_iam_role.github_deploy.arn
  policy_arn    = "arn:${data.aws_partition.current.partition}:eks::aws:cluster-access-policy/AmazonEKSEditPolicy"

  access_scope {
    type       = "namespace"
    namespaces = [var.kubernetes_namespace]
  }

  depends_on = [aws_eks_access_entry.github_deploy]
}

resource "aws_ecr_repository" "component" {
  for_each = var.ecr_repository_names

  name                 = each.value
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_secretsmanager_secret" "database" {
  name                    = "${var.secret_prefix}/database"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "rabbitmq" {
  name                    = "${var.secret_prefix}/rabbitmq"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "application" {
  name                    = "${var.secret_prefix}/application"
  recovery_window_in_days = 7
}

data "aws_iam_policy_document" "pod_identity_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole", "sts:TagSession"]

    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "workload" {
  name               = "nguven-demo-workload"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_trust.json
}

data "aws_iam_policy_document" "workload_secrets" {
  statement {
    sid     = "ReadOnlyNGuvenRuntimeSecrets"
    effect  = "Allow"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.database.arn,
      aws_secretsmanager_secret.rabbitmq.arn,
      aws_secretsmanager_secret.application.arn
    ]
  }
}

resource "aws_iam_role_policy" "workload_secrets" {
  name   = "nguven-demo-runtime-secrets"
  role   = aws_iam_role.workload.id
  policy = data.aws_iam_policy_document.workload_secrets.json
}

resource "aws_eks_pod_identity_association" "workload" {
  cluster_name    = var.eks_cluster_name
  namespace       = var.kubernetes_namespace
  service_account = var.kubernetes_service_account
  role_arn        = aws_iam_role.workload.arn
}
