module "voxel_galileo_bucket" {
  providers = {
    aws = aws
  }
  source                     = "../shared/s3-bucket"
  target_account_id          = var.target_account_id
  primary_region             = var.primary_region
  bucket_name                = "voxel-galileo"
}