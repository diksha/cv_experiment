import boto3


def create_session_with_role(
    role_arn: str, session_name: str
) -> boto3.Session:
    """Assume the given role and return a boto3 session with that role

    Args:
        role_arn (str): arn for an aws IAM role
            example: "arn:aws:iam::360054435465:role/developer_access_assumable_role"
        session_name (str): name to give to the new session

    Returns:
        boto3.Session: session with assumed role
    """
    if role_arn is None:
        return None

    res = boto3.client("sts").assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )

    return boto3.Session(
        aws_access_key_id=res["Credentials"]["AccessKeyId"],
        aws_secret_access_key=res["Credentials"]["SecretAccessKey"],
        aws_session_token=res["Credentials"]["SessionToken"],
    )


def assume_role_for_default_session(role_arn: str):
    """Assume provided role and set the default boto session to use that role

    Args:
        role_arn (str): arn for an aws IAM role
            example: "arn:aws:iam::360054435465:role/developer_access_assumable_role"
    """
    if role_arn is not None:
        res = boto3.client("sts").assume_role(
            RoleArn=role_arn,
            RoleSessionName="production_graph",
        )

        boto3.setup_default_session(
            aws_access_key_id=res["Credentials"]["AccessKeyId"],
            aws_secret_access_key=res["Credentials"]["SecretAccessKey"],
            aws_session_token=res["Credentials"]["SessionToken"],
        )
