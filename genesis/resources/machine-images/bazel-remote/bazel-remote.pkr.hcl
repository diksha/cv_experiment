packer {
    required_plugins {
        amazon = {
            version = ">= 0.0.2"
            source = "github.com/hashicorp/amazon"
        }
    }
}

source "amazon-ebs" "amzn2-arm64" {
    ami_name = "voxel-bazel-remote-0.0.2"
    instance_type = "im4gn.large"
    region = "us-west-2"
    source_ami_filter {
        filters = {
            architecture = "arm64"
            name = "*amzn2-ami-hvm-*"
            root-device-type = "ebs"
        }
        most_recent = true
        owners = ["137112412989"]
    }
    ssh_username = "ec2-user"
}

build {
    name = "bazel-remote"
    sources = [
        "souirce.amazon-ebs.amzn2-arm64"
    ]
    provisioner "file" {
        source = "./files"
        destination = "/tmp/files"
    }
    provisioner "shell" {
        script = "provision.sh"
    }
}