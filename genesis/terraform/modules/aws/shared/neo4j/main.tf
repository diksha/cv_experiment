resource "random_password" "password" {
  length           = 20
  special          = true
  override_special = "_"
}

resource "helm_release" "primary" {
  name             = "${var.db_identifier}-neo4j"
  namespace        = var.db_identifier
  chart            = "https://github.com/neo4j/helm-charts/releases/download/5.2.0/neo4j-5.2.0.tgz"
  wait             = var.wait_for_db
  timeout          = 300
  replace          = false
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    })
  ]
  set_sensitive {
    name = "neo4j.password"
    value = random_password.password.result
  }  
  set {
    name = "neo4j.name"
    value = var.db_identifier
  }
}