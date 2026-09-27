# Evaluation pipeline

CDK app for the steering-visualization image evaluation pipeline. See
`../docs/superpowers/specs/2026-09-26-evaluation-pipeline-design.md` for the
full design.

## Setup

```bash
cd evaluation
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## Tests

```bash
.venv/bin/pytest tests/unit/ -v
```

## Prerequisites (one-time, manual, cannot be done via CDK)

- **Bedrock model access**: in the target account, go to the Bedrock console
  in `us-east-1` -> **Model access** and request access to `GPT-5.6 Terra`
  and `GPT-5.6 Luna` (provider OpenAI). This is an account-level entitlement
  that CloudFormation/CDK cannot grant. Verify it is active before the first
  test upload:
  ```bash
  aws bedrock-runtime converse --profile <your-profile> --region us-east-1 \
    --model-id us.openai.gpt-5.6-terra \
    --messages '[{"role":"user","content":[{"text":"say hi"}]}]' \
    --inference-config '{"maxTokens":5}'
  ```
  An `AccessDeniedException` here means access is not granted yet; a normal
  response means it is. Repeat for `us.openai.gpt-5.6-luna`.
- **CDK bootstrap**: run once per account/region (see Deploy below) if not
  already done -- check with
  `aws cloudformation describe-stacks --stack-name CDKToolkit --region us-east-1`.
- **Deploying IAM identity**: needs permission to create S3 buckets, Lambda
  functions and IAM roles, Glue databases/tables, an Athena workgroup, and SQS
  queues (broad, e.g. `AdministratorAccess`, or an equivalent scoped policy).

## Deploy

The stack is environment-agnostic (no account ID anywhere in this repo) --
whichever AWS credentials/profile you use decide the destination account.

```bash
export AWS_PROFILE=<your-local-profile>
npx aws-cdk@2 bootstrap   # first time only, per account/region
npx aws-cdk@2 deploy --app "$(pwd)/.venv/bin/python app.py"
```

## Manual verification after deploy

Bedrock calls cost money, so this is not automated in CI.

1. In the S3 console, upload an image to `data/layer_0/Animals/Giraffe.png`
   (any photo of a giraffe works -- the label `Giraffe` and category `Animals`
   come from the key itself).
2. Wait roughly a minute, then check
   `results/layer_0/Animals/Giraffe/stringent/recognition/` and
   `results/layer_0/Animals/Giraffe/lenient/recognition/` -- each should have
   10 files (`recognition_1.txt` .. `recognition_10.txt`).
3. Shortly after, check the sibling `eval/` folders under `stringent/` and
   `lenient/` -- each should have 10 files (`eval_1.txt` .. `eval_10.txt`),
   each containing a single `0` or `1`.
4. Run a smoke query in Athena (workgroup `steering-visualization`):
   ```sql
   SELECT category, label, prompt_type, output
   FROM steering_visualization.recognition_results
   WHERE layer = '0' AND category = 'Animals' AND label = 'Giraffe';
   ```
   Expect 20 rows (2 prompt types x 10 rounds), each `output` a single word.
   Repeat against `steering_visualization.evaluation_results` for the parsed
   `0`/`1` judgments.
5. Delete `data/layer_0/Animals/Giraffe.png` from the S3 console. After the
   `cleanup` Lambda runs, confirm the `results/layer_0/Animals/Giraffe/` tree
   is gone, and re-run the Athena queries from step 4 -- both should now
   return zero rows.
