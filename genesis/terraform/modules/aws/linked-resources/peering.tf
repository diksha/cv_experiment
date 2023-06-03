module "vpc_peering_root_rds_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.root
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = "vpc-0ef3dc5eb0ff9e8cb"
  auto_accept_peering = true
}

module "vpc_peering_production_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.production
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = data.aws_vpc.production_perception.id
  auto_accept_peering = true
}

module "vpc_peering_production_portal_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.production
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = data.aws_vpc.production_portal.id
  auto_accept_peering = true
}


module "vpc_peering_root_with_production_product" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.root
    aws.peer = aws.production
  }

  this_vpc_id         = "vpc-0ef3dc5eb0ff9e8cb"
  peer_vpc_id         = data.aws_vpc.production_portal.id
  auto_accept_peering = true
}


module "vpc_peering_production_sales_marketing_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.production
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = data.aws_vpc.production_sales_marketing.id
  auto_accept_peering = true
}

module "vpc_peering_staging_portal_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.staging
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = data.aws_vpc.staging_portal.id
  auto_accept_peering = true
}


# module "vpc_peering_development_with_platform" {
#   source  = "grem11n/vpc-peering/aws"
#   version = ">= 4.1.0"
#   providers = {
#     aws.this = aws.platform
#     aws.peer = aws.development
#   }

#   this_vpc_id         = data.aws_vpc.platform_primary.id
#   peer_vpc_id         = data.aws_vpc.development_perception.id
#   auto_accept_peering = true
# }

module "vpc_peering_prime_with_platform" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.platform
    aws.peer = aws.prime
  }

  this_vpc_id         = data.aws_vpc.platform_primary.id
  peer_vpc_id         = data.aws_vpc.prime_devops.id
  auto_accept_peering = true
}


module "vpc_peering_galileo_with_prime" {
  source  = "grem11n/vpc-peering/aws"
  version = ">= 4.1.0"
  providers = {
    aws.this = aws.prime
    aws.peer = aws.galileo
  }

  this_vpc_id         = data.aws_vpc.prime_devops.id
  peer_vpc_id         = data.aws_vpc.galileo_galileo.id
  auto_accept_peering = true
}

data "aws_vpc" "production_perception" {
  provider = aws.production
  tags = {
    "Name" = "perception"
  }
}

# data "aws_vpc" "staging_perception" {
#   provider = aws.staging
#   tags = {
#     "Name" = "perception"
#   }
# }

# data "aws_vpc" "development_perception" {
#   provider = aws.development
#   tags = {
#     "Name" = "perception"
#   }
# }


data "aws_vpc" "prime_devops" {
  provider = aws.prime
  tags = {
    "Name" = "devops"
  }
}

data "aws_vpc" "platform_primary" {
  provider = aws.platform
  tags = {
    "Name" = "primary"
  }
}

data "aws_vpc" "production_portal" {
  provider = aws.production
  tags = {
    "Name" = "product"
  }
}

data "aws_vpc" "production_sales_marketing" {
  provider = aws.production
  tags = {
    "Name" = "sales-marketing"
  }
}

data "aws_vpc" "staging_portal" {
  provider = aws.staging
  tags = {
    "Name" = "product"
  }
}

data "aws_vpc" "galileo_galileo" {
  provider = aws.galileo
  tags = {
    "Name" = "galileo"
  }
}