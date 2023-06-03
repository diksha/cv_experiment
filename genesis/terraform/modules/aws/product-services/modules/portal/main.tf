locals {
  service_name                               = "portal"
  full_name                                  = "${local.service_name}-api"
  argo_base_name                             = "${local.service_name}-${var.context.environment}-api"
  secret_manager_name                        = local.service_name
  google_service_account_Secret_Manager_Name = "${local.service_name}/google_service_account_json"
  secret_provider_class_name                 = "${local.full_name}-secret-provider-class"
  portal_release_name                        = "${local.service_name}-api"
  target_namespace                           = "${local.service_name}-api"
  argo_cluster_name                          = "${var.context.environment}-${var.context.eks_cluster_name}"
  environment                                = var.context.environment
  client_id                                  = jsondecode(data.aws_secretsmanager_secret_version.auth0_secrets.secret_string)["AUTH0_MANAGEMENT_CLIENT_ID"]
  client_secret                              = jsondecode(data.aws_secretsmanager_secret_version.auth0_secrets.secret_string)["AUTH0_MANAGEMENT_CLIENT_SECRET"]
  domains = {
    "production" : "voxelprod.us.auth0.com"
    "staging" : "voxelstaging.us.auth0.com"
  }
  mumbai_region = "ap-south-1"
}

resource "random_uuid" "cloudfront_token" {
}

resource "aws_ecr_repository" "api" {
  name = "${local.environment}/${local.service_name}/api"
}

resource "aws_ecr_lifecycle_policy" "lifecycle_policy" {
  repository = aws_ecr_repository.api.name

  policy = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 50 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 50
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}


resource "aws_ecr_repository_policy" "api_policy" {
  repository = aws_ecr_repository.api.name

  policy = <<EOF
{
    "Version": "2008-10-17",
    "Statement": [
        {
            "Sid": "AllowBuildkitePushPolicy",
            "Effect": "Allow",
            "Principal": {
              "AWS": "arn:aws:iam::203670452561:role/BuildkiteAccess"
            },
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload"
            ]
        }
    ]
}
EOF
}

resource "aws_s3_bucket" "static_web" {
  bucket = "voxel-${local.service_name}-${local.environment}-static-resources"

  tags = {
    Name        = "voxel-${local.service_name}-${local.environment}-static-resources"
    Environment = "${local.environment}"
  }
}

resource "aws_s3_bucket_ownership_controls" "static_web_disable_acl" {
  bucket = aws_s3_bucket.static_web.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_web_encryption" {
  bucket = aws_s3_bucket.static_web.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "static_web_versioning" {
  bucket = aws_s3_bucket.static_web.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "static_web_access_block" {
  bucket = aws_s3_bucket.static_web.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_policy" "sm_policy" {
  name        = "secret-manager-read-policy"
  path        = "/"
  description = "the policy for allowing pods to read from secret manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}

data "aws_ip_ranges" "cloudfront_global" {
  services = ["cloudfront"]
}

module "irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = local.target_namespace
  serviceaccount = local.full_name
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.context.target_account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = [
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    resource.aws_iam_policy.sm_policy.arn,
  ]
}

resource "kubernetes_service_account" "sa" {
  automount_service_account_token = false
  metadata {
    name      = local.full_name
    namespace = local.target_namespace
    annotations = {
      "eks.amazonaws.com/role-arn" : module.irsa.arn
      "meta.helm.sh/release-name"      = local.secret_provider_class_name
      "meta.helm.sh/release-namespace" = local.full_name
    }
    labels = {
      "app.kubernetes.io/instance"   = local.secret_provider_class_name
      "app.kubernetes.io/managed-by" = "Helm"
      "app.kubernetes.io/name"       = "SecretProviderClass"
      "app.kubernetes.io/version"    = "1.16.0"
      "helm.sh/chart"                = "SecretProviderClass-0.1.0"
    }

  }
}


resource "helm_release" "SecretProviderClass" {
  depends_on = [
    module.irsa
  ]
  count            = 1
  atomic           = true
  cleanup_on_fail  = true
  name             = "${local.full_name}-secret-provider-class"
  namespace        = local.target_namespace
  chart            = "${path.module}/../../../shared/charts/SecretProviderClass"
  version          = "0.1.0"
  wait             = false
  timeout          = 300
  replace          = false
  create_namespace = false
  force_update     = true
  values = [templatefile("${path.module}/files/portal-api-secret-provider-class-values.yaml",
    {
      IAM_ROLE_ARN         = module.irsa.arn
      NAME                 = local.full_name
      SM_NAME              = local.secret_manager_name
      GSA_SM_NAME          = local.google_service_account_Secret_Manager_Name
      SERVICE_ACCOUNT_NAME = local.full_name
      ENVIRONMENT          = local.environment
    })
  ]
}

resource "argocd_application" "portal_migration_job" {
  metadata {
    name      = "${local.argo_base_name}-migrate"
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = "portal-${local.environment}"
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "main"
      helm {
        parameter {
          name  = "image.tag"
          value = var.starting_image_tag
        }

        values = templatefile("${path.module}/files/portal-api-migrate-job-values.yaml",
          {
            NAME                 = "${local.full_name}-migrate"
            SERVICE_ACCOUNT_NAME = local.full_name
            IAM_ROLE_ARN         = module.irsa.arn
            PORTAL_REPO          = aws_ecr_repository.api.repository_url
            PORTAL_IMAGE_TAG     = var.starting_image_tag
            SECRET_PROVIDER_NAME = local.secret_provider_class_name
            ENVIRONMENT          = local.environment
            DOMAIN               = var.domain
        })

        release_name = "${local.full_name}-migrate"
      }
    }

    destination {
      name      = local.argo_cluster_name
      namespace = local.target_namespace
    }
    sync_policy {
      automated = {
        allow_empty = false
        prune       = true
        self_heal   = true
      }
      sync_options = ["CreateNamespace=true", "Replace=true"]
      retry {
        limit = "5"
        backoff = {
          duration     = "30s"
          max_duration = "2m"
          factor       = "2"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      spec[0].source[0].helm[0].parameter
    ]
  }
}

resource "argocd_application" "portal_api" {
  metadata {
    name      = local.argo_base_name
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = "portal-${local.environment}"
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "3ed6f055e804f6025a048d5d16ddc92f48289d52"
      helm {
        parameter {
          name         = "command"
          value        = "/app/core/portal/serve"
          force_string = false
        }
        parameter {
          name         = "image.tag"
          value        = var.starting_image_tag
          force_string = false
        }

        # value_files  = ["values-test.yml"]
        values = templatefile("${path.module}/files/portal-api-service-values.yaml",
          {
            NAME                 = local.full_name
            SERVICE_ACCOUNT_NAME = local.full_name
            PORTAL_REPO          = aws_ecr_repository.api.repository_url
            PORTAL_IMAGE_TAG     = var.starting_image_tag
            SECRET_PROVIDER_NAME = local.secret_provider_class_name
            ENVIRONMENT          = local.environment
            ACM_ARN              = aws_acm_certificate.cert_origin.arn
            CLOUDFRONT_IP_RANGES = join(",", data.aws_ip_ranges.cloudfront_global.cidr_blocks)
            SERVICE_NAME         = "${local.full_name}-generic"
            DOMAIN               = var.domain
            CLOUDFRONT_TOKEN     = random_uuid.cloudfront_token.result
        })

        release_name = local.full_name
      }
    }

    destination {
      name = local.argo_cluster_name

      namespace = local.target_namespace
    }
    sync_policy {
      automated = {
        allow_empty = false
        prune       = true
        self_heal   = true
      }
      sync_options = ["CreateNamespace=true", "Replace=true"]
      retry {
        limit = "5"
        backoff = {
          duration     = "30s"
          max_duration = "2m"
          factor       = "2"
        }
      }
    }
  }
  lifecycle {
    ignore_changes = [
      spec[0].source[0].helm[0].parameter
    ]
  }

}

resource "argocd_application" "portal_api_state_message_worker_service" {
  metadata {
    name      = "${local.argo_base_name}-state-message-worker"
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = "portal-${local.environment}"
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "main"
      helm {
        parameter {
          name  = "image.tag"
          value = var.starting_image_tag
        }


        values = templatefile("${path.module}/files/portal-workers-service-values.yaml",
          {
            NAME                 = "${local.full_name}-state-message-worker"
            SERVICE_ACCOUNT_NAME = local.full_name
            PORTAL_REPO          = aws_ecr_repository.api.repository_url
            PORTAL_IMAGE_TAG     = var.starting_image_tag
            SECRET_PROVIDER_NAME = local.secret_provider_class_name
            ENVIRONMENT          = local.environment
            COMMAND_ARG          = "--message_type=state"
            DOMAIN               = var.domain
        })

        release_name = "${local.full_name}-state-message-worker"
      }
    }

    destination {
      name = local.argo_cluster_name

      namespace = local.target_namespace
    }
    sync_policy {
      automated = {
        allow_empty = false
        prune       = true
        self_heal   = true
      }
      sync_options = ["CreateNamespace=true", "Replace=true"]
      retry {
        limit = "5"
        backoff = {
          duration     = "30s"
          max_duration = "2m"
          factor       = "2"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      spec[0].source[0].helm[0].parameter
    ]
  }

}

resource "argocd_application" "portal_api_event_message_worker_service" {
  metadata {
    name      = "${local.argo_base_name}-event-message-worker"
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = "portal-${local.environment}"
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "main"
      helm {
        parameter {
          name  = "image.tag"
          value = var.starting_image_tag
        }

        values = templatefile("${path.module}/files/portal-workers-service-values.yaml",
          {
            NAME                 = "${local.full_name}-event-message-worker"
            SERVICE_ACCOUNT_NAME = local.full_name
            PORTAL_REPO          = aws_ecr_repository.api.repository_url
            PORTAL_IMAGE_TAG     = var.starting_image_tag
            SECRET_PROVIDER_NAME = local.secret_provider_class_name
            ENVIRONMENT          = local.environment
            COMMAND_ARG          = "--message_type=event"
            DOMAIN               = var.domain
        })

        release_name = "${local.full_name}-event-message-worker"
      }
    }

    destination {
      name      = local.argo_cluster_name
      namespace = local.target_namespace
    }
    sync_policy {
      automated = {
        allow_empty = false
        prune       = true
        self_heal   = true
      }
      sync_options = ["CreateNamespace=true", "Replace=true"]
      retry {
        limit = "5"
        backoff = {
          duration     = "30s"
          max_duration = "2m"
          factor       = "2"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      spec[0].source[0].helm[0].parameter
    ]
  }

}

# Ingress should be defined outside argo, such that any force recreate in argo doesn't repalces ALB.
# Unhealthy threshold to 10 to get around 1 min of health checks as pods can take time to get ready.
resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name      = local.full_name
    namespace = local.target_namespace
    annotations = {
      "alb.ingress.kubernetes.io/scheme"                                = "internet-facing"
      "alb.ingress.kubernetes.io/certificate-arn"                       = aws_acm_certificate.cert_origin.arn
      "alb.ingress.kubernetes.io/target-type"                           = "ip"
      "alb.ingress.kubernetes.io/actions.ssl-redirect"                  = "{\"Type\": \"redirect\", \"RedirectConfig\": { \"Protocol\": \"HTTPS\", \"Port\": \"443\", \"StatusCode\": \"HTTP_301\"}}"
      "alb.ingress.kubernetes.io/conditions.${local.full_name}-generic" = "[{\"field\":\"http-header\",\"httpHeaderConfig\":{\"httpHeaderName\": \"X-CLOUDFRONT-TOKEN\", \"values\":[\"${random_uuid.cloudfront_token.result}\"]}}]"
      "alb.ingress.kubernetes.io/healthcheck-path"                      = "/api/health"
      "alb.ingress.kubernetes.io/unhealthy-threshold-count"             = "10"
      "alb.ingress.kubernetes.io/target-group-attributes"               = "deregistration_delay.timeout_seconds=120"
    }
  }

  spec {
    ingress_class_name = "alb"
    rule {
      host = var.domain
      http {
        path {
          backend {
            service {
              name = "${local.full_name}-generic"
              port {
                number = 80
              }
            }
          }
          path      = "/"
          path_type = "Prefix"
        }
      }
    }
    tls {
      hosts       = [var.domain]
      secret_name = "${local.full_name}-generic-tls"
    }
  }
}


module "cdn" {
  source                         = "terraform-aws-modules/cloudfront/aws"
  aliases                        = [var.domain]
  comment                        = "portal-cdn"
  enabled                        = true
  is_ipv6_enabled                = true
  price_class                    = "PriceClass_100"
  default_root_object            = "static/frontend/index.html"
  web_acl_id                     = aws_wafv2_web_acl.portal_web_waf.arn
  retain_on_delete               = false
  wait_for_deployment            = false
  create_monitoring_subscription = true
  create_origin_access_identity  = false
  origin = {
    service-origin = {
      domain_name = kubernetes_ingress_v1.ingress.status[0].load_balancer[0].ingress[0].hostname
      custom_origin_config = {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }

      custom_header = [
        {
          name  = "X-CLOUDFRONT-TOKEN"
          value = random_uuid.cloudfront_token.result
        }
      ]
    }

    s3-static = {
      domain_name              = aws_s3_bucket.static_web.bucket_regional_domain_name
      origin_access_control_id = aws_cloudfront_origin_access_control.s3_static_origin_access_control.id
    }

    s3-frontend = {
      domain_name              = aws_s3_bucket.static_web.bucket_regional_domain_name
      origin_path              = "/static/frontend"
      origin_access_control_id = aws_cloudfront_origin_access_control.s3_static_origin_access_control.id
    }
  }

  default_cache_behavior = {
    target_origin_id       = "s3-frontend"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    use_forwarded_values   = false

    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
    cache_policy_id            = "658327ea-f89d-4fab-a63d-7e88639e58f6"

    function_association = {

      viewer-request = {
        function_arn = aws_cloudfront_function.request_handler.arn
        include_body = true
      }
    }
  }

  ordered_cache_behavior = [
    {
      path_pattern               = "/api/*"
      target_origin_id           = "service-origin"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      origin_request_policy_id   = aws_cloudfront_origin_request_policy.portal_service_request.id
      allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = aws_cloudfront_cache_policy.no_cache.id
      use_forwarded_values       = false
    },
    {
      path_pattern               = "/graphql*"
      target_origin_id           = "service-origin"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      origin_request_policy_id   = aws_cloudfront_origin_request_policy.portal_service_request.id
      allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = aws_cloudfront_cache_policy.no_cache.id
      use_forwarded_values       = false

    },
    {
      path_pattern               = "/admin**"
      target_origin_id           = "service-origin"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      origin_request_policy_id   = aws_cloudfront_origin_request_policy.portal_service_request.id
      allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = aws_cloudfront_cache_policy.no_cache.id
      use_forwarded_values       = false

    },
    {
      path_pattern               = "/static/admin/*"
      target_origin_id           = "s3-static"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      allowed_methods            = ["GET", "HEAD"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = "658327ea-f89d-4fab-a63d-7e88639e58f6"
      use_forwarded_values       = false
    },
    {
      path_pattern               = "/static/graphene_django/*"
      target_origin_id           = "s3-static"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      allowed_methods            = ["GET", "HEAD"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = "658327ea-f89d-4fab-a63d-7e88639e58f6"
      use_forwarded_values       = false
    },
    {
      path_pattern               = "/internal/backend/*"
      target_origin_id           = "service-origin"
      viewer_protocol_policy     = "redirect-to-https"
      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_header_policy.id
      origin_request_policy_id   = aws_cloudfront_origin_request_policy.portal_service_request.id
      allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = aws_cloudfront_cache_policy.no_cache.id
      use_forwarded_values       = false
    }
  ]

  viewer_certificate = {
    acm_certificate_arn      = aws_acm_certificate.cert.arn
    minimum_protocol_version = "TLSv1.2_2021"
    ssl_support_method       = "sni-only"
  }
}


resource "aws_cloudfront_response_headers_policy" "security_header_policy" {
  name = "portal-custom-security-header"
  security_headers_config {
    content_type_options {
      override = true
    }
    frame_options {
      frame_option = "SAMEORIGIN"
      override     = false
    }
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = false
    }
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = false
      override                   = false
      preload                    = false
    }
    xss_protection {
      mode_block = true
      override   = false
      protection = true
    }
  }
}

resource "aws_cloudfront_function" "request_handler" {
  name    = "cloudfront-spa-request-handler"
  runtime = "cloudfront-js-1.0"
  code    = <<EOF
function handler(event) {
  var request = event.request;
  var path = request.uri.split(/\#|\?/)[0];
  if (path.split(".").length === 1) {
    // Request URI is not a static asset path, rewrite URI to SPA entrypoint
    request.uri = "/index.html";
  }
  return request;
}
EOF
}

resource "aws_cloudfront_cache_policy" "no_cache" {
  name        = "no-cache-policy"
  default_ttl = 0
  max_ttl     = 0
  min_ttl     = 0
  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "none"
    }
    query_strings_config {
      query_string_behavior = "none"
    }
  }
}

resource "aws_cloudfront_origin_request_policy" "portal_service_request" {
  name    = "portal-service-request-policy"
  comment = "Forward relevant headers and cookies to service origins"
  cookies_config {
    cookie_behavior = "all"
  }
  headers_config {
    header_behavior = "allViewer"
  }
  query_strings_config {
    query_string_behavior = "all"
  }
}

resource "aws_acm_certificate" "cert" {
  provider = aws.east

  domain_name       = var.domain
  validation_method = "DNS"

  tags = {
    Environment = local.environment
    Service     = var.name
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_acm_certificate" "cert_origin" {
  domain_name       = var.domain
  validation_method = "DNS"

  tags = {
    Environment = local.environment
    Service     = var.name
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudfront_origin_access_control" "s3_static_origin_access_control" {
  name                              = "s3_static_origin_access_control"
  description                       = "S3 Static Origin Access Control Policy"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}


data "aws_iam_policy_document" "static_web_s3_policy" {
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = ["${aws_s3_bucket.static_web.arn}/*", "${aws_s3_bucket.static_web.arn}"]
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [module.cdn.cloudfront_distribution_arn]
    }
  }
  statement {
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/BuildkiteAccess"]
    }
    actions = [
      "s3:ListBucket",
      "s3:ListMultipartUploadParts",
      "s3:ListBucketMultipartUploads",
      "s3:GetObject",
      "s3:GetObjectAcl",
      "s3:PutObject",
      "s3:PutObjectAcl",
      "s3:AbortMultipartUpload",
      "s3:DeleteObject"
    ]
    resources = [
      aws_s3_bucket.static_web.arn,
      "${aws_s3_bucket.static_web.arn}/*",
    ]
  }
}

resource "aws_s3_bucket_policy" "static_web" {
  bucket = aws_s3_bucket.static_web.id
  policy = data.aws_iam_policy_document.static_web_s3_policy.json
}

resource "aws_wafv2_web_acl" "portal_web_waf" {
  name     = "portal-web-waf"
  scope    = "CLOUDFRONT"
  provider = aws.east

  default_action {
    allow {}
  }

  rule {
    name     = "AWS-AWSManagedRulesCommonRuleSet"
    priority = 0

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWS-AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "portal-web-waf"
    sampled_requests_enabled   = true
  }
}

resource "kubernetes_labels" "pod_readiness_gate_label" {
  api_version = "v1"
  kind        = "Namespace"
  metadata {
    name      = local.target_namespace
    namespace = local.target_namespace
  }
  labels = {
    "elbv2.k8s.aws/pod-readiness-gate-inject" = "enabled"
  }
}

data "aws_secretsmanager_secret_version" "auth0_secrets" {
  secret_id = "portal"
}

resource "auth0_rule" "trigger_mfa_enrollment" {
  name    = "trigger-mfa"
  enabled = true
  order   = 1
  script  = file("${path.module}/files/trigger-mfa-enrollment.js")
}

resource "auth0_rule" "add_user_metadata" {
  name    = "add-user-metadata-to-request"
  enabled = true
  order   = 2
  script  = file("${path.module}/files/add-user-metadata.js")
}

resource "auth0_attack_protection" "portal_protection" {
  suspicious_ip_throttling {
    enabled = true
    shields = ["admin_notification", "block"]

    pre_login {
      max_attempts = 100
      rate         = 864000
    }

    pre_user_registration {
      max_attempts = 50
      rate         = 1200
    }
  }

  brute_force_protection {
    enabled      = true
    max_attempts = 5
    mode         = "count_per_identifier_and_ip"
    shields      = ["block", "user_notification"]
  }
}

resource "aws_iam_role" "s3_signing_role" {
  name = "s3-signing-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = module.irsa.arn
        }
      }
    ]
  })
}
resource "aws_iam_role_policy" "s3_signing_role_policy" {
  name = "s3-signing-role-policy"
  role = aws_iam_role.s3_signing_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject"
        ]
        Effect   = "Allow"
        Resource = [
          "${var.voxel_portal_bucket_arn}/*"
        ]
      }
    ]
  })
}


module "mumbai_bucket" {
  primary_region = local.mumbai_region
  source            = "../../../shared/s3-bucket"
  target_account_id = var.context.target_account_id
  bucket_name       = "voxel-portal-${local.environment}-mumbai"
  expiration_days   = 7
  enable_versioning = true

  providers = {
    aws = aws.mumbai
  }
}

resource "aws_s3_bucket_policy" "mumbai_bucket_policy" {
  provider = aws.mumbai

  bucket = module.mumbai_bucket.bucket_id
  policy = jsonencode({
    Statement = [
      {
        Sid = "ReadWrite Permission"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation",
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-portal-${local.environment}-mumbai",
          "arn:aws:s3:::voxel-portal-${local.environment}-mumbai/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/BuildkiteAccess",
          ]
        }
      },
      {
        Sid = "Read Only Permission"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-portal-${local.environment}-mumbai",
          "arn:aws:s3:::voxel-portal-${local.environment}-mumbai/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6",
            # sematic-worker irsa for EKS Pod in Prime Account DevOps VPC Jenkins Cluster
            "arn:aws:iam::203670452561:role/irsa-tt6b0fvxadvn",
          ]
        }
      }
    ]
    Version = "2012-10-17"
  })

}

resource "aws_s3_bucket_cors_configuration" "mumbai_bucket_cors" {
  provider = aws.mumbai

  bucket = module.mumbai_bucket.bucket_id
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = [local.environment == "production" ? "https://app.voxelai.com" : "https://app.staging.voxelplatform.com"]
    expose_headers  = ["Access-Control-Allow-Origin"]
    max_age_seconds = 3000
  }
}


resource "aws_s3control_multi_region_access_point" "access_point" {
  details {
    name = "portal-access-point"

    region {
      bucket = var.voxel_bucket_portal_name
    }

    region {
      bucket = module.mumbai_bucket.bucket_name
    }
  }
}

resource "aws_s3_bucket_replication_configuration" "rep_configuration" {
  bucket = var.voxel_bucket_portal_name
  role   = aws_iam_role.replication_role.arn
  rule {
    id = "multipoint-rep-rule"
    status = "Enabled"

    destination {
      bucket = module.mumbai_bucket.bucket_arn
    }
  }
}

resource "aws_iam_role" "replication_role" {
  name = "s3-bucket-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "replication_policy" {
  name   = "s3-bucket-replication-policy"
  role   = aws_iam_role.replication_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags",
          "s3:GetObjectVersionTagging",
          "s3:PutObject"
        ]
        Effect   = "Allow"
        Resource = ["${module.mumbai_bucket.bucket_arn}/*", module.mumbai_bucket.bucket_arn]
      },
      {
        Action = [
          "s3:GetObject*",
          "s3:GetBucketVersioning",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = ["${var.voxel_portal_bucket_arn}/*", var.voxel_portal_bucket_arn]
      }
    ]
  })
}

resource "aws_s3control_multi_region_access_point_policy" "access_point_policy" {
  details {
    name = element(split(":", aws_s3control_multi_region_access_point.access_point.id), 1)
    policy =  jsonencode({
    Statement = [
      {
        Sid = "Read Only Permission"
        Action = [
          "s3:GetObject"
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3::${var.context.target_account_id}:accesspoint/${aws_s3control_multi_region_access_point.access_point.alias}/object/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6",
            # sematic-worker irsa for EKS Pod in Prime Account DevOps VPC Jenkins Cluster
            "arn:aws:iam::203670452561:role/irsa-tt6b0fvxadvn",
          ]
        }
      }
    ]
    Version = "2012-10-17"
  })
  }
}
