locals {
    domain = "*.galileo.private.voxelplatform.com"
}

resource "aws_acm_certificate" "cert_origin" {
  domain_name       = local.domain
  validation_method = "DNS"

  tags = {
    Service     = "galileo"
  }

  lifecycle {
    create_before_destroy = true
  }
}