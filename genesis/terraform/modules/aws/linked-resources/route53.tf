resource "aws_route53_zone" "voxelplatform_private" {
  provider = aws.platform

  name = "test.voxelplatform.com"

  vpc {
    vpc_id = data.aws_vpc.platform_primary.id
  }

  # Prevent the deletion of associated VPCs after
  # the initial creation. See documentation on
  # aws_route53_zone_association for details
  lifecycle {
    ignore_changes = [vpc]
  }
}

# Root Account
resource "aws_route53_vpc_association_authorization" "root_account_vpc" {
  provider = aws.platform
  vpc_id  = "vpc-0ef3dc5eb0ff9e8cb"
  zone_id = aws_route53_zone.voxelplatform_private.id
}

resource "aws_route53_zone_association" "root_account_vpc" {
  provider = aws.root
  vpc_id  = aws_route53_vpc_association_authorization.root_account_vpc.vpc_id
  zone_id = aws_route53_vpc_association_authorization.root_account_vpc.zone_id
}

# Prime Account
resource "aws_route53_vpc_association_authorization" "prime_account_devops_vpc" {
  provider = aws.platform
  vpc_id  = data.aws_vpc.prime_devops.id
  zone_id = aws_route53_zone.voxelplatform_private.id
}

resource "aws_route53_zone_association" "prime_account_devops_vpc" {
  provider = aws.prime
  vpc_id  = aws_route53_vpc_association_authorization.prime_account_devops_vpc.vpc_id
  zone_id = aws_route53_vpc_association_authorization.prime_account_devops_vpc.zone_id
}

# Production Account
resource "aws_route53_vpc_association_authorization" "production_account_perception_vpc" {
  provider = aws.platform
  vpc_id  = data.aws_vpc.production_perception.id
  zone_id = aws_route53_zone.voxelplatform_private.id
}

resource "aws_route53_zone_association" "production_account_perception_vpc" {
  provider = aws.production
  vpc_id  = aws_route53_vpc_association_authorization.production_account_perception_vpc.vpc_id
  zone_id = aws_route53_vpc_association_authorization.production_account_perception_vpc.zone_id
}

resource "aws_route53_vpc_association_authorization" "production_account_product_vpc" {
  provider = aws.platform
  vpc_id  = data.aws_vpc.production_portal.id
  zone_id = aws_route53_zone.voxelplatform_private.id
}

resource "aws_route53_zone_association" "production_account_product_vpc" {
  provider = aws.production
  vpc_id  = aws_route53_vpc_association_authorization.production_account_product_vpc.vpc_id
  zone_id = aws_route53_vpc_association_authorization.production_account_product_vpc.zone_id
}

# Staging Account
resource "aws_route53_vpc_association_authorization" "staging_account_product_vpc" {
  provider = aws.platform
  vpc_id  = data.aws_vpc.staging_portal.id
  zone_id = aws_route53_zone.voxelplatform_private.id
}

resource "aws_route53_zone_association" "staging_account_product_vpc" {
  provider = aws.staging
  vpc_id  = aws_route53_vpc_association_authorization.staging_account_product_vpc.vpc_id
  zone_id = aws_route53_vpc_association_authorization.staging_account_product_vpc.zone_id
}

# resource "aws_route53_vpc_association_authorization" "staging_account_perception_vpc" {
#   provider = aws.platform
#   vpc_id  = data.aws_vpc.staging_perception.id
#   zone_id = aws_route53_zone.voxelplatform_private.id
# }

# resource "aws_route53_zone_association" "staging_account_perception_vpc" {
#   provider = aws.staging
#   vpc_id  = aws_route53_vpc_association_authorization.staging_account_perception_vpc.vpc_id
#   zone_id = aws_route53_vpc_association_authorization.staging_account_perception_vpc.zone_id
# }


resource "aws_route53_record" "wildcard_private_voxelplatform_com" {
  provider = aws.platform
  zone_id = aws_route53_zone.voxelplatform_private.zone_id
  name    = "*.private"
  type    = "CNAME"
  ttl     = 5
  records = ["k8s-ingressn-ingressn-df39accd4a-eba6f015c5bed84d.elb.us-west-2.amazonaws.com"]
}
