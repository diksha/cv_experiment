output cert_pem {
    value = base64decode(shell_script.intermediate_ca.output["cert"])
}

output key_pem {
    value = base64decode(shell_script.intermediate_ca.output["key"])
    sensitive = true
}
