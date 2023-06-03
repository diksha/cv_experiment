resource "aws_acm_certificate" "primary" {
  provider = aws.certificate
  tags = {
    "DOMAIN" = var.domain
  }

  domain_name       = "*.${var.domain}"
  subject_alternative_names = var.include_domain_in_san ? [var.domain]: []
  validation_method = "DNS"
}

resource "aws_route53_record" "primary_certificates" {
  provider = aws.route53
  for_each = {
    for dvo in aws_acm_certificate.primary.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name

  records = [each.value.record]
  ttl     = 1800
  type    = each.value.type
  zone_id = var.route53_zone_id
}

resource "aws_acm_certificate_validation" "primary_validation" {
  provider                = aws.certificate
  certificate_arn         = aws_acm_certificate.primary.arn
  validation_record_fqdns = [for record in aws_route53_record.primary_certificates : record.fqdn]
}
