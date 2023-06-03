variable "target_account_id" {
  type = string
}

variable "primary_region" {
  type = string
}


variable "bucket_name" {
  type = string
}


variable "noncurrent_days" {
  default = 90
}

variable "expiration_days" {
  default = 0
}

variable "enable_versioning" {
  default = true
}

variable "enable_intelligent_tiering" {
  default = true
}

variable "additional_tags" {
  default     = {}
  description = "Additional resource tags"
  type        = map(string)
}
