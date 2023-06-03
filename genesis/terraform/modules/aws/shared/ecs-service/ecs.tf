resource "aws_ecs_task_definition" "task_definition" {
  family             = "${var.service_identifier}_service"
  cpu                = var.cpu_millicores
  memory             = var.memory_in_mb
  execution_role_arn = aws_iam_role.service.arn
  container_definitions = jsonencode([
    {
      name        = var.service_identifier
      image       = var.image_name
      entrypoint  = var.entrypoint
      essential   = true
      environment = var.environment_vars
      portMappings = [
        {
          containerPort = var.container_port
          # hostPort      = var.container_port
        }
      ]
      secrets = [
        for index, secret_var in var.secret_vars : {
          name      = secret_var.name
          valueFrom = aws_secretsmanager_secret_version.secrets[secret_var.key].arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/${var.service_identifier}-container"
          awslogs-region        = var.primary_region
          awslogs-create-group  = "true"
          awslogs-stream-prefix = var.service_identifier
          mode                  = "non-blocking"
        }
      }
      "healthCheck" : {
        "retries" : var.healthcheck_retries,
        "command" : [
          "CMD-SHELL",
          " (! command -v curl &>/dev/null) || curl -X ${var.healthcheck_method} -f http://localhost:${var.container_port}/${var.healthcheck_subpath}"
        ],
        "timeout" : var.healthcheck_timeout,
        "interval" : var.healthcheck_interval,
        "startPeriod" : var.healthcheck_start_period
      },
    }
  ])
  network_mode             = var.launch_type == "EC2" ? "bridge" : "awsvpc"
  requires_compatibilities = [var.launch_type]
}

resource "aws_ecs_service" "service" {
  name                               = "${var.service_identifier}_service"
  cluster                            = var.ecs_cluster_id
  task_definition                    = aws_ecs_task_definition.task_definition.arn
  desired_count                      = var.desired_count
  launch_type                        = var.launch_type
  force_new_deployment               = var.dev_mode
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 200
  load_balancer {
    target_group_arn = module.alb.target_group_arns[0]
    container_name   = var.service_identifier
    container_port   = var.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  dynamic "ordered_placement_strategy" {
    for_each = var.launch_type == "EC2" ? ["blah"] : []
    content {
      type  = "binpack"
      field = "memory"
    }
  }
  dynamic "network_configuration" {
    for_each = var.launch_type == "EC2" ? [] : ["blah"]
    content {
      subnets          = var.svc_subnet_ids
      security_groups  = var.svc_security_group_ids
      assign_public_ip = false
    }
  }
}
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_scaling_capacity
  min_capacity       = var.min_scaling_capacity
  resource_id        = "service/${var.ecs_cluster_id}/${aws_ecs_service.service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "memory_based_scaling" {
  name               = "${aws_ecs_service.service.name}-memory"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }

    target_value = 75
  }
}


resource "aws_appautoscaling_policy" "cpu_based_scaling" {
  name               = "${aws_ecs_service.service.name}-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    target_value = 75
  }
}
