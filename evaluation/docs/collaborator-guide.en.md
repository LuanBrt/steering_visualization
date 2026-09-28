# Collaborator guide — steering-visualization evaluation pipeline

*(Portuguese original: [`collaborator-guide.md`](collaborator-guide.md))*

You've been given access to upload test images, watch the automatic results
come in, and query everything via SQL. You don't need to know anything else
about AWS beyond what's here.

## 1. Setting up your access

You should have received from the account owner: a username/password (or an
access key), the **Account ID**, and the role name
(`steering-visualization-collaborator`). In the examples below,
`<ACCOUNT_ID>` is that number you were given (it is not written into this
guide on purpose, since this guide is public).

### Option A — AWS Console (easiest for exploring/querying)

1. Sign in at `https://<ACCOUNT_ID>.signin.aws.amazon.com/console` with your username and password.
2. In the top-right menu, click **Switch Role** (or go directly to
   https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-console.html
   if you can't find the button).
3. Account: `<ACCOUNT_ID>`. Role: `steering-visualization-collaborator`.
   Give it a display name if you like (e.g. "steering-viz").

After that, the console will already take you to S3 and Athena with the
right access.

### Option B — AWS CLI (easiest for batch uploads / scripts)

Set up your own credentials and a profile that assumes the role
automatically (see
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html):

```ini
# ~/.aws/config
[profile alice]
aws_access_key_id = <your access key id>
aws_secret_access_key = <your secret access key>
region = us-east-1

[profile steering-visualization]
role_arn = arn:aws:iam::<ACCOUNT_ID>:role/steering-visualization-collaborator
source_profile = alice
region = us-east-1
```

From then on, every command uses `--profile steering-visualization` and the
AWS CLI assumes the role for you behind the scenes:

```bash
aws s3 ls s3://steering-visualization/ --profile steering-visualization
```

## 2. S3 path format (how to name your images)

Every image goes under:

```
data/layer_<N>/<category>/<label>/<variant>.<ext>
```

- **`layer`**: the layer number of the model whose steering generated the image.
- **`category`**: a simple English word/phrase (it gets injected literally
  into a sentence like `"What {category} is in the image..."`, so avoid odd
  names).
- **`label`**: the concept the image was supposed to represent (e.g. `Giraffe`).
- **`variant`**: identifies which example image this is, among possibly
  several for the same `label` (e.g. `0`, `1`, `2`, ...). If you only have
  one image for that concept, use `0` anyway.
- **`ext`**: `png`, `jpg`, `jpeg`, `gif`, or `webp`.

### Example — Giraffe

Uploading the first example image of a giraffe, at layer 0:

```bash
aws s3 cp Giraffe.png \
  s3://steering-visualization/data/layer_0/Animals/Giraffe/0.png \
  --profile steering-visualization
```

A second example image of the same concept (useful to check whether the
result is consistent across different images of the same concept):

```bash
aws s3 cp AnotherGiraffe.png \
  s3://steering-visualization/data/layer_0/Animals/Giraffe/1.png \
  --profile steering-visualization
```

Each one triggers processing automatically and independently -- you don't
need to wait for one to finish before uploading the other.

**Uploaded the wrong name?** Just delete it (`aws s3 rm ...`) and re-upload
with the right name -- cleaning up the old results happens automatically.

## 3. What happens automatically

Within ~1 minute (it can take longer depending on load), 40 files appear
under:

```
results/layer_<N>/<category>/<label>/<variant>/stringent/recognition/recognition_1..10.txt
results/layer_<N>/<category>/<label>/<variant>/stringent/eval/eval_1..10.txt
results/layer_<N>/<category>/<label>/<variant>/lenient/recognition/recognition_1..10.txt
results/layer_<N>/<category>/<label>/<variant>/lenient/eval/eval_1..10.txt
```

- `recognition_N.txt` = what the model answered (free text, usually one word).
- `eval_N.txt` = `0` or `1` -- whether a second model judged the answer as mentioning the expected concept.
- `stringent` = a neutral question ("What is in the image?"); `lenient` = a question guided by the category ("What Animals is in the image if you had to guess?").
- 10 rounds of each, to smooth out the model's variability.

## 4. Viewing the results

### Downloading directly from S3

```bash
# a single file
aws s3 cp s3://steering-visualization/results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt - \
  --profile steering-visualization

# an entire image's folder
aws s3 sync s3://steering-visualization/results/layer_0/Animals/Giraffe/0/ ./giraffe-0-results/ \
  --profile steering-visualization
```

### Querying via Athena (recommended for analysis)

Open the Athena console (workgroup **`steering-visualization`**, database
**`steering_visualization`**) -- see
https://docs.aws.amazon.com/athena/latest/ug/querying-athena-tables.html
if this is your first time using the Athena query editor.

Two tables: `recognition_results` (column `output`) and
`evaluation_results` (column `evaluation`, `"0"` or `"1"`). Both partitioned
by `layer`, `category`, `label`, `variant`, `prompt_type`.

**See the raw responses for one image:**
```sql
SELECT variant, prompt_type, output
FROM steering_visualization.recognition_results
WHERE layer = '0' AND category = 'Animals' AND label = 'Giraffe'
ORDER BY variant, prompt_type;
```

**Success rate per example image (variant), to compare different images of the same concept:**
```sql
SELECT variant, prompt_type, count(*) AS rounds, sum(cast(evaluation AS integer)) AS successes
FROM steering_visualization.evaluation_results
WHERE layer = '0' AND label = 'Giraffe'
GROUP BY variant, prompt_type
ORDER BY variant, prompt_type;
```

**Success rate aggregated by concept, across all layers:**
```sql
SELECT category, label, prompt_type, avg(cast(evaluation AS double)) AS success_rate
FROM steering_visualization.evaluation_results
GROUP BY category, label, prompt_type
ORDER BY category, label, prompt_type;
```

**Extracting the round number and joining recognition + eval for the same round** (the round number is not a direct column -- it comes from the filename via the `"$path"` pseudo-column):
```sql
SELECT category, label, variant, prompt_type,
       regexp_extract("$path", 'recognition_(\d+)\.txt', 1) AS round,
       output
FROM steering_visualization.recognition_results
WHERE layer = '0' AND label = 'Giraffe';
```

## Useful links

- Switching roles in the console: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-console.html
- Setting up a CLI profile that assumes a role automatically: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html
- Running queries in the Athena editor: https://docs.aws.amazon.com/athena/latest/ug/querying-athena-tables.html
- S3 CLI command reference: https://docs.aws.amazon.com/cli/latest/reference/s3/
