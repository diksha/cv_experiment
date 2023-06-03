variable "root_account_id" {
  type = string
}

variable "primary_region" {
  type = string
}

variable "production" {
  type = map
}

variable "staging" {
  type = map
}

variable "development" {
  type = map
}

variable "platform" {
  type = map
}

variable "prime" {
  type = map
}

variable "galileo" {
  type = map
}

variable "aws_notifications_slack_app_webhook_url" {
  type      = string
  sensitive = true
}