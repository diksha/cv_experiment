output bucket_id {
    value = resource.aws_s3_bucket.bucket.id
}

output bucket_arn {
    value = resource.aws_s3_bucket.bucket.arn
}

output bucket_name {
    value = var.bucket_name
}