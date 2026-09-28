# Evaluation pipeline

CDK app for the steering-visualization image evaluation pipeline. See
`../docs/superpowers/specs/2026-09-26-evaluation-pipeline-design.en.md` for
the full design (Portuguese original:
`../docs/superpowers/specs/2026-09-26-evaluation-pipeline-design.md`).

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

- **Bedrock model access**: the code currently uses Claude Sonnet 4.5
  (recognition) and Claude Haiku 4.5 (evaluation) -- see the "Model
  substitution" note below. These were available on this account without
  any extra request. If you want to try the originally-intended OpenAI
  models instead (`openai.gpt-5.6-terra` / `openai.gpt-5.6-luna`, see design
  spec section 4), request access in the Bedrock console (`us-east-1` ->
  **Model access**) and verify with:
  ```bash
  aws bedrock-runtime converse --profile <your-profile> --region us-east-1 \
    --model-id us.openai.gpt-5.6-terra \
    --messages '[{"role":"user","content":[{"text":"say hi"}]}]' \
    --inference-config '{"maxTokens":5}'
  ```
  An `AccessDeniedException` here means access is not granted; a normal
  response means it is. On this account, that call kept failing for hours
  after the model agreement showed as accepted -- do not assume it will work
  quickly.
- **CDK bootstrap**: run once per account/region (see Deploy below) if not
  already done -- check with
  `aws cloudformation describe-stacks --stack-name CDKToolkit --region us-east-1`.
- **Deploying IAM identity**: needs permission to create S3 buckets, Lambda
  functions and IAM roles, Glue databases/tables, an Athena workgroup, and SQS
  queues (broad, e.g. `AdministratorAccess`, or an equivalent scoped policy).

## Model substitution (read this before comparing results to the paper)

The original paper uses GPT-5 (recognition) and GPT-5-mini (evaluation) via
the OpenAI API. This pipeline targeted the closest Bedrock equivalents
(`openai.gpt-5.6-terra` / `openai.gpt-5.6-luna`), but access to them was
never authorized on this AWS account, even after accepting the model
agreement (see the prerequisite above). The pipeline currently runs on
**Claude Sonnet 4.5** (recognition) and **Claude Haiku 4.5** (evaluation)
instead -- marked `PROVISIONAL` in `steering_evaluation/evaluation_stack.py`,
with the original OpenAI model IDs commented alongside for an easy revert.
Any recognition rate or confidence interval produced by this pipeline
reflects Claude's behavior, not GPT-5's -- treat comparisons to the paper's
published numbers as qualitative, not exact.

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

1. In the S3 console, upload an image to `data/layer_0/Animals/Giraffe/0.png`
   (any photo of a giraffe works -- `category=Animals`, `label=Giraffe`,
   `variant=0` all come from the key itself; see design spec section 3 for
   the full path format, including why `variant` exists).
2. Wait roughly a minute, then check
   `results/layer_0/Animals/Giraffe/0/stringent/recognition/` and
   `results/layer_0/Animals/Giraffe/0/lenient/recognition/` -- each should
   have 10 files (`recognition_1.txt` .. `recognition_10.txt`).
3. Shortly after, check the sibling `eval/` folders under `stringent/` and
   `lenient/` -- each should have 10 files (`eval_1.txt` .. `eval_10.txt`),
   each containing a single `0` or `1`.
4. Run a smoke query in Athena (workgroup `steering-visualization`):
   ```sql
   SELECT category, label, variant, prompt_type, output
   FROM steering_visualization.recognition_results
   WHERE layer = '0' AND category = 'Animals' AND label = 'Giraffe';
   ```
   Expect 20 rows (2 prompt types x 10 rounds), each `output` a single word.
   Repeat against `steering_visualization.evaluation_results` for the parsed
   `0`/`1` judgments.
5. Delete `data/layer_0/Animals/Giraffe/0.png` from the S3 console. After the
   `cleanup` Lambda runs, confirm the `results/layer_0/Animals/Giraffe/0/`
   tree is gone, and re-run the Athena queries from step 4 -- both should now
   return zero rows.

## Generating a recognition-rate report

Once results exist in Athena, `evaluation/scripts/generate_recognition_report.py`
writes one CSV per prompt type (stringent/lenient) with Wilson and Wald 95%
confidence intervals per concept/layer, and
`evaluation/scripts/plot_recognition_rate.py` turns either CSV into a small-multiples
chart styled after the paper's Figure 3 (one panel per concept, shaded CI band):

```bash
.venv/bin/python scripts/generate_recognition_report.py --profile <your-profile>
python3 scripts/plot_recognition_rate.py \
  --csv reports/recognition_rate_by_layer_lenient.csv \
  --output reports/recognition_rate_by_layer_lenient.png \
  --ci wald
```

`scripts/upload_experiment_results.py` bridges the local image-generation
script's raw output (`results/word_experiment_vqgan_gemma/<category>/<label>/layer-<N>/`)
into this pipeline's expected `data/` layout, uploading only the `best.png`
checkpoint as `variant=0` and skipping anything already uploaded -- safe to
re-run whenever new results land locally (`--dry-run` to preview first).
