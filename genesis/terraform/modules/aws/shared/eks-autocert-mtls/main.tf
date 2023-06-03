locals {
  root_ca_cert_pem = var.root_ca.cert_pem
  root_ca_key_pem = var.root_ca.key_pem
  root_ca_password = var.root_ca.password
  
  intermediate_ca_common_name = var.common_name
  intermediate_ca_password = random_password.intermediate_ca_password.result
  intermediate_ca_key_pem = module.intermediate_ca.key_pem
  intermediate_ca_cert_pem = module.intermediate_ca.cert_pem

  provisioner_password = random_password.provisioner_password.result
  provisioner_name = "voxel-services"
  provisioner_json = module.ca_provisioner.raw_json

  dns_name = "step-certificates.default.svc.cluster.local"

  step_certificates_values = sensitive(templatefile("${path.module}/files/step-certificates-values.yaml", {
    PROVISIONER_JSON = local.provisioner_json,
    INTERMEDIATE_CA_KEY = local.intermediate_ca_key_pem,
    INTERMEDIATE_CA_CERT = local.intermediate_ca_cert_pem,
    ROOT_CA_CERT = local.root_ca_cert_pem,
    DNS_NAME = local.dns_name,
  }))

  autocert_values = sensitive(templatefile("${path.module}/files/autocert-values.yaml", {
    DNS_NAME = local.dns_name,
    PROVISIONER_NAME = local.provisioner_name,
    ROOT_CA_CERT_PEM = local.root_ca_cert_pem,
  }))
}

resource "random_password" "intermediate_ca_password" {
    length = 16
    special = false
}

resource "random_password" "provisioner_password" {
    length = 16
    special = false
}

resource "helm_release" "step_certificates" {
  name       = "step-certificates"
  repository = "https://smallstep.github.io/helm-charts/"
  chart      = "step-certificates"
  version    = "v1.23.2"

  values = [local.step_certificates_values]

  # there seems to be some kind of bug with set_sensitive
  # where it does not appear to like setting the intermediate_ca_key
  # value, so unfortunately we have to set it in the template
  # this isn't too terrible since that key is password protected
  set_sensitive {
    name = "inject.secrets.ca_password"
    value = base64encode(local.intermediate_ca_password)
  }

  set_sensitive {
    name = "inject.secrets.provisioner_password"
    value = base64encode(local.provisioner_password)
  }

  lifecycle {
    replace_triggered_by = [null_resource.recreate_trigger]
  }
}

resource "null_resource" "recreate_trigger" {
  triggers = {
    "root_ca" = local.root_ca_cert_pem
    "intermediate_ca" = local.intermediate_ca_cert_pem
    "ca_provisioner" = local.provisioner_json
  }
}

resource "helm_release" "autocert" {
  depends_on = [
    helm_release.step_certificates
  ]
  name = "autocert"
  repository = "https://smallstep.github.io/helm-charts/"
  chart = "autocert"
  version = "v1.17.1"

  values = [local.autocert_values]

  set_sensitive {
    name = "ca.provisioner.password"
    value = local.provisioner_password
  }

  lifecycle {
    replace_triggered_by = [null_resource.recreate_trigger]
  }
}


module "intermediate_ca" {
  source = "./intermediate-ca"
  root_ca = var.root_ca
  common_name = local.intermediate_ca_common_name
  password = local.intermediate_ca_password
}

module "ca_provisioner" {
  source = "./ca-provisioner"
  name = local.provisioner_name
  password = local.provisioner_password
}