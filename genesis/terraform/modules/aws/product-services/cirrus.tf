module "cirrus" {
  source            = "./modules/cirrus-service"
  name              = "cirrus"
  context           = var.context
  initial_image_tag = "tritonserver:22.07-py3"
  oidc_provider     = local.oidc_provider
  command_args      = ["--model_repository", "s3://{module.cirrus-models-bucket.bucket_name}/model-repo", "model_control_mode", "poll"] #TODO: make changes to add triton server names here
}

#S3 bucket
module "cirrus-models-bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = local.target_account_id
  primary_region    = local.primary_region
  bucket_name       = "voxel-${local.environment}-triton-models"
  enable_versioning = true
  noncurrent_days   = 90
}