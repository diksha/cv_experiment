locals {
  daily_backup_tag = { "${local.s3_daily_backup_tag_key}" : "${local.s3_daily_backup_tag_value}" }
}

module "voxel_consumable_labels_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-consumable-labels"
  enable_versioning = true
  noncurrent_days   = 365
  additional_tags   = local.daily_backup_tag
}

module "voxel_datasets_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-datasets"
  enable_versioning = true
  noncurrent_days   = 90
}

module "voxel_logs_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-logs"
  enable_versioning = true
  noncurrent_days   = 365
  additional_tags   = local.daily_backup_tag
}

module "voxel_raw_labels_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-raw-labels"
  enable_versioning = true
  noncurrent_days   = 90
  additional_tags   = local.daily_backup_tag
}

module "voxel_raw_logs_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-raw-logs"
  enable_versioning = true
  noncurrent_days   = 90
}

module "voxel_temp_bucket" {
  source                     = "../shared/s3-bucket"
  target_account_id          = var.target_account_id
  primary_region             = var.primary_region
  bucket_name                = "voxel-temp"
  enable_versioning          = false
  enable_intelligent_tiering = false
  expiration_days            = 7
}

module "voxel_temp_dataops_bucket" {
  source                     = "../shared/s3-bucket"
  target_account_id          = var.target_account_id
  primary_region             = var.primary_region
  bucket_name                = "voxel-temp-dataops"
  enable_versioning          = false
  enable_intelligent_tiering = false
  expiration_days            = 7
}

module "voxel_users_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-users"
  enable_versioning = false
}

module "voxel_lightly_input_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-lightly-input"
  enable_versioning = false
  additional_tags   = local.daily_backup_tag
}

module "voxel_lightly_output_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-lightly-output"
  enable_versioning = false
  additional_tags   = local.daily_backup_tag
}

module "voxel_portal_dev_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-portal-dev"
  enable_versioning = false
}

resource "aws_s3_bucket_cors_configuration" "voxel_portal_dev_bucket_cors" {
  bucket = module.voxel_portal_dev_bucket.bucket_id
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = ["http://localhost:9000"]
    expose_headers  = ["Access-Control-Allow-Origin"]
    max_age_seconds = 3000
  }
}

module "voxel_storage_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-storage"
  enable_versioning = true
  noncurrent_days   = 365
  additional_tags   = local.daily_backup_tag
}

module "voxel_models_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-models"
  enable_versioning = true
  noncurrent_days   = 365
  additional_tags   = local.daily_backup_tag
}

module "voxel_perception_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-perception"
  enable_versioning = true
  noncurrent_days   = 365
}

module "voxel_metaverse_backup_bucket" {
  source                     = "../shared/s3-bucket"
  target_account_id          = var.target_account_id
  primary_region             = var.primary_region
  bucket_name                = "voxel-metaverse-backup"
  enable_versioning          = false
  enable_intelligent_tiering = false
  expiration_days            = 30
}

module "voxel_infinity_ai_shared_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-infinity-ai-shared"
  enable_versioning = true
  noncurrent_days   = 365
  additional_tags   = local.daily_backup_tag
}
