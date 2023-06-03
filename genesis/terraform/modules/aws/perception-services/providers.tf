provider "aws" {
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.target_account_id}:role/TerraformAccess"
  }
}

provider "argocd" {
  server_addr = "${var.argo_server_domain}:${var.argo_server_port}"
  username    = var.argo_username
  password    = var.argo_password
  insecure    = true
}

data "aws_eks_cluster" "cluster" {
  name = var.eks_cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.eks_cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}