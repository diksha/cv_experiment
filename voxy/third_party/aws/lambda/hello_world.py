import boto3
from loguru import logger


def lambda_handler(*args):
    """Sample lambda handler

    Args:
        args (list): lambda payload
    """
    logger.info("Hello, world!")
    logger.info(args)
    logger.info(boto3.client("sts").get_caller_identity())


if __name__ == "__main__":
    lambda_handler()
