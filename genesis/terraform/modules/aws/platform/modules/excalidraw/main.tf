resource "aws_ecr_repository" "excalidraw" {
  name = "excalidraw"
  image_tag_mutability = "IMMUTABLE"

}

resource "aws_ecr_lifecycle_policy" "ecr_policy" {
  repository = aws_ecr_repository.excalidraw.name
  policy = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 5 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 5
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}

resource "helm_release" "primary" {
  name             = var.deplomyment_identifier
  namespace        = var.deplomyment_identifier
  chart            = "${path.module}/../../../shared/charts/generic"
  wait             = true
  timeout          = 300
  replace          = false
  create_namespace = true
  values = [
    templatefile("${path.module}/files/service-helm-values.yaml",
      {
        REPOSITORY = "448273619612.dkr.ecr.us-west-2.amazonaws.com/excalidraw"
        TAG  = "2660d7722a53"
    })
  ]
}

resource "helm_release" "collab" {
  name             = "${var.deplomyment_identifier}-collab"
  namespace        = var.deplomyment_identifier
  chart            = "${path.module}/../../../shared/charts/generic"
  wait             = true
  timeout          = 300
  replace          = false
  create_namespace = true
 values = [
    templatefile("${path.module}/files/service-helm-values.yaml",
      {
        REPOSITORY = "excalidraw/excalidraw-room"
        TAG  = "sha-c0bf0ba"
    })
  ]
}

resource "kubernetes_ingress_v1" "excalidraw_ingress" {
  metadata {
    name      = "excalidraw"
    namespace = "excalidraw"
    annotations = {
      "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3600"
      "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600"
      "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600"
    }
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "draw.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "excalidraw-generic"
              port {
                number = 80
              }
            }
          }
          path = "/"
        }
      }
    }
  }
}



