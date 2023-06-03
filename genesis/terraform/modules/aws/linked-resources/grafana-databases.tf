resource "grafana_data_source" "portal_production_postgres" {
  type          = "postgres"
  name          = "portal-production-postgres"
  url           = "portal-production-postgres.c1trxszive18.us-west-2.rds.amazonaws.com"
  database_name = "portal_production_postgres"
  username      = "grafana"

  is_default = false
  json_data {
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }

  secure_json_data {
    password = var.portal_production_postgres_pwd
  }
}

resource "grafana_data_source" "portal_production_postgres_reviewers_org" {
  provider      = grafana.reviewers
  type          = "postgres"
  name          = "portal-production-postgres-reviewers"
  url           = "portal-production-postgres.c1trxszive18.us-west-2.rds.amazonaws.com"
  database_name = "portal_production_postgres"
  username      = "grafana"

  is_default = false
  json_data {
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }

  secure_json_data {
    password = var.portal_production_postgres_pwd
  }
}

resource "grafana_data_source" "portal_production_timescale" {
  type          = "postgres"
  name          = "portal-production-timescale"
  url           = "state-and-events-development-voxel.a.timescaledb.io:11689"
  database_name = "production"
  username      = "tsdbadmin"
  is_default    = false

  json_data {
    timescaledb              = true
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }
  secure_json_data {
    password = var.portal_production_timescale_pwd
  }
}

resource "grafana_data_source" "portal_staging_postgres" {
  type          = "postgres"
  name          = "portal-staging-postgres"
  url           = "portal-staging-db.cluster-ccidhb6kjfj1.us-west-2.rds.amazonaws.com"
  database_name = "portal-staging"
  username      = "app"
  is_default    = false

  json_data {
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }
  secure_json_data {
    password = var.portal_staging_postgres_pwd
  }
}


resource "grafana_data_source" "postgres_perception_scenario_eval" {
  type          = "postgres"
  name          = "postgres-perception-scenario-eval"
  url           = "34.83.174.121"
  database_name = "metrics"
  username      = "grafana"
  is_default    = false

  json_data {
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }

  secure_json_data {
    password = var.postgres_perception_scenario_eval_password
  }
}


resource "grafana_data_source" "sales_marketing_production_postgres" {
  type          = "postgres"
  name          = "postgres-sales-marketing"
  url           = "10.66.222.3"
  database_name = "postgres"
  username      = "grafana"
  is_default    = false

  json_data {
    postgres_version         = 1300
    ssl_mode                 = "require"
    tls_auth                 = true
    tls_auth_with_ca_cert    = true
    tls_configuration_method = "file-path"
  }

  secure_json_data {
    password = var.sales_marketing_production_postgres_pwd
  }
}

