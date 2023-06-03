# trunk-ignore-all(pylint/E0611): pylint thinks `sendgrid` is this module
# trunk-ignore-all(pylint/W0406): pylint thinks `sendgrid` is this module
from typing import List, Optional, Tuple, Union

from django.conf import settings
from loguru import logger
from python_http_client import exceptions
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Asm,
    Bcc,
    Content,
    From,
    GroupId,
    GroupsToDisplay,
    Mail,
)

# Template IDs
ASSIGN_EMAIL_TEMPLATE_ID = "d-2c5d7b04e1c94e33a0a1910e97afa7db"
INCIDENT_RESOLVED_TEMPLATE_ID = "d-3f72395fa02e46c981202d717fdb7e58"
DAILY_SUMMARY_TEMPLATE_ID = "d-99c4a77ca1124fcbaf512fbfc4309f17"
USER_INVITATION_TEMPLATE_ID = "d-bf7353d8bf1a4b5bb864480306f728f3"
INCIDENT_ALERT_TEMPLATE_ID = "d-e26377fa5fb94efbab0e2350791b98b3"
ZONE_PULSE_TEMPLATE_ID = "d-b47f11c20b5849a1b3f2024129a2b81e"

# Advanced Suppression Manager (ASM) group IDs
DAILY_SUMMARY_ASM_GROUP_ID = 17387
INCIDENT_ALERT_ASM_GROUP_ID = 18639
ZONE_PULSE_ASM_GROUP_ID = 18717


class SendGridClient:
    def __init__(self):
        if not settings.SENDGRID_API_KEY:
            raise RuntimeError("Sendgrid API key is null.")
        self.sg = SendGridAPIClient(settings.SENDGRID_API_KEY)

    def send_email(
        self,
        from_email: str,
        to_emails: Union[str, List[str]],
        subject: str,
        plain_content: str = None,
        html_content: str = None,
    ):
        """Sends email"""

        from_email = From(from_email)

        mail = Mail(from_email, to_emails, subject)

        if plain_content:
            mail.add_content(Content("text/plain", plain_content))

        if html_content:
            mail.add_content(Content("text/html", html_content))

        self._send(mail)

    def send_email_with_template(
        self,
        *,
        from_email: str = None,
        to_emails: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        bcc: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        subject: str = None,
        template_id: str = None,
        asm_group_id: int = None,
        **kwargs
    ):
        """Sends email using sendgrid transactional template"""

        if not from_email:
            raise RuntimeError("from_email is required to send email.")

        if not to_emails:
            raise RuntimeError(
                "At least one email must be specified via to_emails to send email."
            )

        data = dict(kwargs)
        data.update(subject=subject)

        mail = Mail(
            from_email=From(from_email),
            to_emails=to_emails,
            subject=subject,
        )
        mail.template_id = template_id
        mail.dynamic_template_data = data

        if isinstance(bcc, list):
            for x in bcc:
                mail.add_bcc(Bcc(x))
        elif isinstance(bcc, str):
            mail.add_bcc(Bcc(bcc))

        # Advanced Suppression Manager (configured in SendGrid dashboard)
        # Required for users to unsubscribe from subsets of messages
        # instead of unsubscribing to all of our emails globally.
        if asm_group_id:
            mail.asm = Asm(
                GroupId(asm_group_id), GroupsToDisplay([asm_group_id])
            )

        self._send(mail)

    def _send(self, mail):
        if not settings.SEND_TRANSACTIONAL_EMAILS:
            logger.warning(
                "Not sending email because SEND_TRANSACTIONAL_EMAILS is False"
            )
            logger.warning(mail.get())
            return

        try:
            logger.info(mail.get())
            _ = self.sg.send(mail)
        except exceptions.BadRequestsError as e:
            logger.error("An error occurred during sending email")
            logger.error(e.body)
