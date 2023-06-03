provider "aws" {
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.target_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
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

data "aws_eks_cluster" "cluster" {
  name = module.base-account-setup.eks_cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.base-account-setup.eks_cluster_name
}
