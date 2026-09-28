# Inviting colleagues to the evaluation pipeline (admin guide)

*(Portuguese original: [`admin-invite-collaborators.md`](admin-invite-collaborators.md))*

This guide is for you (the AWS account owner). It shows how to give a
colleague access to upload images, query Athena, and download results --
without giving them access to anything else in the account.

The CDK already created a **role** (`steering-visualization-collaborator`)
with exactly the permissions needed. No one can assume it until you
explicitly authorize each user -- this is done once per colleague, outside
the CDK (on purpose: the list of who has access should not require an
infrastructure redeploy every time someone joins or leaves the team).

Find your deployment account's Account ID (we do not write this number into
any repository file, on purpose -- see spec section 12):
```bash
aws sts get-caller-identity --profile <your-profile> --query Account --output text
```
In the commands below, `<ACCOUNT_ID>` is that number.

- Role ARN: `arn:aws:iam::<ACCOUNT_ID>:role/steering-visualization-collaborator`
- Sessions last up to 12h (the maximum allowed) -- your colleague doesn't need to reassume the role all the time.

## Step 0 (one time only): create the group that authorizes assuming the role

```bash
aws iam create-group --group-name steering-visualization-collaborators

aws iam put-group-policy \
  --group-name steering-visualization-collaborators \
  --policy-name AssumeCollaboratorRole \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::<ACCOUNT_ID>:role/steering-visualization-collaborator"
    }]
  }'
```

Any IAM user placed in this group will be able to assume the role. You only
need to do this once; the remaining steps are per colleague.

## Step 1 (per colleague): create the IAM user

```bash
aws iam create-user --user-name alice
aws iam add-user-to-group --group-name steering-visualization-collaborators --user-name alice
```

## Step 2: give them a way to authenticate

Pick one (or both):

**AWS console access** (good if they'll use Athena/S3 through the web UI):
```bash
aws iam create-login-profile \
  --user-name alice \
  --password 'ChangeThisPassword123!' \
  --password-reset-required
```
Send them: the **Account ID** (the number from above), the username
(`alice`), and this temporary password -- they change it on first login at
`https://<ACCOUNT_ID>.signin.aws.amazon.com/console`.

**Access key for the command line** (good if they'll use the AWS CLI):
```bash
aws iam create-access-key --user-name alice
```
This prints `AccessKeyId` and `SecretAccessKey` -- send them over a secure
channel (not plain-text email/Slack). This is the only time the
`SecretAccessKey` is shown.

## Step 3: send them

- The **colleague's guide**: `evaluation/docs/collaborator-guide.md`
- The **Account ID**: the number you got at the start of this guide
- The **role name**: `steering-visualization-collaborator`
- The Step 2 credentials

## Removing someone's access

```bash
aws iam remove-user-from-group --group-name steering-visualization-collaborators --user-name alice
# optional: delete the user entirely
aws iam delete-login-profile --user-name alice          # if you created a console password
aws iam list-access-keys --user-name alice               # to find the access key id
aws iam delete-access-key --user-name alice --access-key-id <ID>
aws iam delete-user --user-name alice
```

Removing them from the group is already enough to revoke access to the
role -- you don't need to delete the user if it's just a temporary pause.
