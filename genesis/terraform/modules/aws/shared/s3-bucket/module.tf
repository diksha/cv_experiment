resource "aws_s3_bucket" "bucket" {
  bucket = var.bucket_name
  tags = var.additional_tags  
}

resource "aws_s3_bucket_ownership_controls" "disable_acl" {
  bucket = aws_s3_bucket.bucket.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "block_public_access" {
  bucket                  = aws_s3_bucket.bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "versioning" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "lifecycle_rules" {
  count = (var.noncurrent_days > 0 && var.enable_versioning) || var.expiration_days > 0 ? 1 : 0 
  bucket = aws_s3_bucket.bucket.id

  dynamic rule {
    for_each = var.expiration_days > 0 ? [1] : []
    content {
        id = "expiration_days_all"
        expiration {
            days = var.expiration_days
        }
        status = "Enabled"
    }
  }

  dynamic rule {
    for_each = var.noncurrent_days > 0 && var.enable_versioning ? [1] : []
    content {
        id = "noncurrent_days_versioning_all"
        noncurrent_version_expiration {
            noncurrent_days = var.noncurrent_days
        }
        status = "Enabled"
    }
  }

  dynamic rule {
    for_each = var.enable_intelligent_tiering ? [1] : []
    content {
      id = "transtion_to_intelligent_tiering"
      transition {
        storage_class = "INTELLIGENT_TIERING"
      }
      noncurrent_version_transition {
        storage_class = "INTELLIGENT_TIERING"
      }
      status = "Enabled"
    }
  }
  depends_on = [aws_s3_bucket_versioning.versioning]
}

resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
