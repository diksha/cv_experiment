provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate  = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

data "aws_eks_cluster" "cluster" {
  name = var.context.eks_cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.context.eks_cluster_name
}


provider "aws" {
  region = "us-west-2"
  assume_role {
    role_arn = "arn:aws:iam::${var.context.target_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "east"
  region = "us-east-1"
  assume_role {
    role_arn = "arn:aws:iam::${var.context.target_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}


provider "argocd" {
  server_addr = "${var.argo.domain}:${var.argo.port}"
  username    = var.argo.username
  password    = var.argo.password
  insecure    = true
}