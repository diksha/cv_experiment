
module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "6.7.0"

  name               = "ecs-${var.service_identifier}"
  load_balancer_type = "application"
  vpc_id             = var.lb_vpc_id
  subnets            = var.lb_subnet_ids
  security_groups    = var.lb_security_group_ids

  target_groups = [
    {
      name                 = var.service_identifier
      backend_protocol     = "HTTP"
      backend_port         = var.container_port
      slow_start           = var.healthcheck_start_period + 30
      target_type          = var.launch_type == "EC2" ? "instance" : "ip"
      deregistration_delay = 40
      # preserve_client_ip   = true
      health_check = {
        path                = "/${var.healthcheck_subpath}"
        timeout             = var.healthcheck_timeout
        unhealthy_threshold = var.healthcheck_retries
        interval            = var.healthcheck_interval
        healthy_threshold   = 2
        matcher             = var.healthcheck_status_code_matcher
        # port                = var.container_port
      }
    }
  ]

  https_listeners = [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = var.certificate_arn
      target_group_index = 0
    }
  ]

  http_tcp_listeners = [
    {
      port        = 80
      protocol    = "HTTP"
      action_type = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      },
      target_group_index = 0

    }
  ]

  tags = {
    Environment = var.environment
  }
}

