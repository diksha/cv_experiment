resource "random_password" "password" {
  length           = 20
  special          = true
  override_special = "_"
}

resource "helm_release" "primary" {
  name             = "${var.db_identifier}-postgres"
  namespace        = var.namespace != "" ? var.namespace : var.db_identifier
  chart            = "postgresql-ha"
  repository       = "https://charts.bitnami.com/bitnami"
  version          = var.chart_version
  wait             = var.wait_for_db
  timeout          = 300
  replace          = false
  create_namespace = var.create_namespace

  set_sensitive {
    name = "global.postgresql.password"
    value = random_password.password.result
  }  
  
  set_sensitive {
    name = "postgresql.password"
    value = random_password.password.result
  }    
  
  set_sensitive {
    name = "postgresql.repmgrPassword"
    value = random_password.password.result
  }    
  
  set_sensitive {
    name = "pgpool.adminPassword"
    value = random_password.password.result
  }  
  
  set {
    name = "global.postgresql.database"
    value = var.db_identifier
  }

  set {
    name = "global.storageClass"
    value = "gp3-retain"
  }

  set {
    name = "initdbScriptsCM"
    value = var.init_db_script_config_map
  }

  set {
    name = "postgresql.replicaCount"
    value = var.replica_count
  }

  set {
    name = "persistence.size"
    value = var.size
  }
}