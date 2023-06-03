module "galileo_vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = var.environment
  vpc_cidr_root      = var.galileo_vpc_cidr_root
  vpc_name           = "galileo"
  enable_nat_gateway = true
}


resource "aws_vpc_endpoint" "s3" {
  vpc_id       = module.galileo_vpc.vpc_id
  service_name = "com.amazonaws.us-west-2.s3"
}
