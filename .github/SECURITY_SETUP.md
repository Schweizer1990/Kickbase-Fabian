# Security setup

This repository is public. Never commit Kickbase or email credentials.

Required GitHub Actions secrets:
- `KICK_USER`
- `KICK_PASS`

Optional email delivery secrets:
- `EMAIL_USER`
- `EMAIL_PASS`

If the optional email secrets are not set, email delivery should be disabled. Sensitive league tables should not be printed to public GitHub Actions logs.
