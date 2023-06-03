locals {
    provisioner_password = var.password
    provisioner_name = var.name
}

resource "shell_script" "ca_provisioner" {
  lifecycle_commands {
    create = file("${path.module}/../scripts/generate-step-ca-provisioner.sh")
    delete = "echo"
  }

  environment = {
    PROVISIONER_PASSWORD_BASE64ENCODED = base64encode(local.provisioner_password)
    PROVISIONER_NAME = local.provisioner_name
    JQ = "${path.module}/../scripts/jq.sh"
    STEP = "${path.module}/../scripts/step.sh"
  }
}
