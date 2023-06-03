module "drata_production" {
  providers = {
    aws = aws.production
  }
  source = "../shared/drata"
}

module "drata_prime" {
  providers = {
    aws = aws.prime
  }
  source = "../shared/drata"
}

module "drata_staging" {
  providers = {
    aws = aws.staging
  }
  source = "../shared/drata"
}

module "drata_platform" {
  providers = {
    aws = aws.platform
  }
  source = "../shared/drata"
}

module "drata_root" {
  providers = {
    aws = aws.root
  }
  source = "../shared/drata"
}