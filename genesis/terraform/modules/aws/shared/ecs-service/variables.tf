variable "dev_mode" {
  default = false
}
variable "service_identifier" {}
variable "image_name" {}
variable "entrypoint" {
  default = null
  
}
variable "ecs_cluster_id" {}
variable "desired_count" {
  default = 1
}
variable "environment_vars" {
  default = [
    {
      name  = "TEST_ENV"
      value = "VARVAL"
    }
  ]
}

variable "environment" {

}

variable "certificate_arn" {}
variable "container_port" {
  type = number
}

variable "lb_vpc_id" {}
variable "lb_subnet_ids" {}
variable "lb_security_group_ids" {}

variable "svc_subnet_ids" {}
variable "svc_security_group_ids" {}

variable "primary_region" {}

variable "secret_vars" {
  default = [
    {
      key   = "sample-key-1"
      name  = "SAMPLE_ENV"
      value = "SAMPLE_VALUE"
    }
  ]
}

variable "healthcheck_method" {
  default = "GET"
}

variable "healthcheck_status_code_matcher" {
  default = "200"
}

variable "healthcheck_subpath" {
  default = ""
}

variable "healthcheck_interval" {
  default = 30
}

variable "healthcheck_start_period" {
  default = 10
}

variable "healthcheck_timeout" {
  default = 5
}


variable "healthcheck_retries" {
  default = 3
}


variable "cpu_millicores" {
  type    = number
  default = 512
}

variable "memory_in_mb" {
  type    = number
  default = 1024
}

variable max_scaling_capacity {
  default = 100
}

variable min_scaling_capacity {
  default = 1
}

variable "launch_type" {
  default = "FARGATE"
  validation {
    condition     = var.launch_type == "EC2" || var.launch_type == "FARGATE"
    error_message = "Can only be EC2/FARGATE."
  }
}