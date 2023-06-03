variable "namespace" {
    type = string
}

variable "name" {
    type = string
}

variable "create_namespace" {
    type = bool
    default = true
}

variable "model_repository" {
    type = string
}

variable "model_control_mode" {
    type = string
}

variable "oidc_provider" {
    type = string
}
