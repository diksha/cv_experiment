variable context {
    type = object({
        environment = string
    })
}

variable extra_policy_arns {
    type = list(string)
    default = []
}

variable vpc {
    type = object({
        enabled = bool
        security_group_ids = list(string)
        subnet_ids = list(string)
    })
    default = {
        enabled = false
        security_group_ids = []
        subnet_ids = []
    }
}

variable function_name {
    type = string
}

variable repository_name {
    type = string

    validation {
        condition = can(regex("^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*\\/)*[a-z0-9]+(?:[._-][a-z0-9]+)*$", var.repository_name))
        error_message = "Invalid repository name. Check that name is all lowercase, does not end in '-' or have '--' in the name."
    }
}

variable iam_role_name_override {
    type = string
    default = ""
}

variable memory_size {
    type = number
    default = 128
}

variable timeout {
    type = number
    default = 3
}
