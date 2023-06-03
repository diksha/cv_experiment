locals {
    root_ca_cert_pem            = var.root_ca.cert_pem
    root_ca_key_pem             = var.root_ca.key_pem
    root_ca_password            = var.root_ca.password
    intermediate_ca_password    = var.password
    intermediate_ca_common_name = var.common_name
}

resource "shell_script" "intermediate_ca" {
  lifecycle_commands {
    create = file("${path.module}/../scripts/generate-intermediate-ca.sh")
    delete = "echo"
  }

  environment = {
    ROOT_CA_CERT_PEM_BASE64ENCODED = base64encode(local.root_ca_cert_pem)
    ROOT_CA_KEY_PEM_BASE64ENCODED = base64encode(local.root_ca_key_pem)
    ROOT_CA_PASSWORD_BASE64ENCODED = base64encode(local.root_ca_password)
    INTERMEDIATE_CA_PASSWORD_BASE64ENCODED = base64encode(local.intermediate_ca_password)
    INTERMEDIATE_CA_COMMON_NAME = local.intermediate_ca_common_name
    JQ = "${path.module}/../scripts/jq.sh"
    STEP = "${path.module}/../scripts/step.sh"
  }
}