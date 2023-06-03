variable root_ca {
    type = object({
        cert_pem = string
        key_pem = string
        password = string
    })
    sensitive = true
}

variable common_name {
    type = string
}