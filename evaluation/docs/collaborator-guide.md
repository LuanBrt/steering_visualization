# Guia do colaborador — pipeline de avaliação steering-visualization

*(Tradução em inglês: [`collaborator-guide.en.md`](collaborator-guide.en.md))*

Você recebeu acesso pra subir imagens de teste, acompanhar os resultados
automáticos e consultar tudo via SQL. Não é preciso saber nada de AWS além do
que está aqui.

## 1. Configurar seu acesso

Você deve ter recebido do dono da conta: um usuário/senha (ou access key), o
**Account ID** e o nome do role (`steering-visualization-collaborator`). Nos
exemplos abaixo, `<ACCOUNT_ID>` é esse número que você recebeu (não é gravado
neste guia de propósito, já que ele é público).

### Opção A — Console AWS (mais fácil pra explorar/consultar)

1. Entre em `https://<ACCOUNT_ID>.signin.aws.amazon.com/console` com seu usuário e senha.
2. No menu do canto superior direito, clique em **Switch Role** (ou acesse
   diretamente https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-console.html
   se não achar o botão).
3. Account: `<ACCOUNT_ID>`. Role: `steering-visualization-collaborator`.
   Dê um nome de exibição se quiser (ex: "steering-viz").

Depois disso, o console já vai te levar a S3 e Athena com o acesso certo.

### Opção B — AWS CLI (mais fácil pra upload em lote / scripts)

Configure suas próprias credenciais e um profile que assume o role
automaticamente (veja
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html):

```ini
# ~/.aws/config
[profile alice]
aws_access_key_id = <seu access key id>
aws_secret_access_key = <seu secret access key>
region = us-east-1

[profile steering-visualization]
role_arn = arn:aws:iam::<ACCOUNT_ID>:role/steering-visualization-collaborator
source_profile = alice
region = us-east-1
```

A partir daí, todo comando usa `--profile steering-visualization` e o AWS
CLI assume o role sozinho por trás dos panos:

```bash
aws s3 ls s3://steering-visualization/ --profile steering-visualization
```

## 2. Formato dos paths no S3 (como nomear suas imagens)

Toda imagem vai em:

```
data/layer_<N>/<categoria>/<label>/<variant>.<ext>
```

- **`layer`**: número da camada do modelo cujo steering gerou a imagem.
- **`categoria`**: uma palavra/frase simples em inglês (ela entra literalmente
  numa frase tipo `"What {categoria} is in the image..."`, então evite nomes
  estranhos).
- **`label`**: o conceito que a imagem devia representar (ex: `Giraffe`).
- **`variant`**: identifica qual imagem-exemplo é essa, entre várias possíveis
  pro mesmo `label` (ex: `0`, `1`, `2`, ...). Se você só tem uma imagem pra
  esse conceito, use `0` mesmo assim.
- **`ext`**: `png`, `jpg`, `jpeg`, `gif` ou `webp`.

### Exemplo — Giraffe

Subindo a primeira imagem-exemplo de uma girafa, na layer 0:

```bash
aws s3 cp Giraffe.png \
  s3://steering-visualization/data/layer_0/Animals/Giraffe/0.png \
  --profile steering-visualization
```

Uma segunda imagem-exemplo do mesmo conceito (útil pra ver se o resultado é
consistente entre imagens diferentes do mesmo conceito):

```bash
aws s3 cp OutraGiraffe.png \
  s3://steering-visualization/data/layer_0/Animals/Giraffe/1.png \
  --profile steering-visualization
```

Cada uma dispara o processamento automaticamente e de forma independente —
não precisa esperar uma terminar pra subir a outra.

**Enviou o nome errado?** Só apagar (`aws s3 rm ...`) e subir de novo com o
nome certo — a limpeza dos resultados antigos é automática.

## 3. O que acontece automaticamente

Em ~1 minuto (pode levar mais, dependendo da carga), aparecem 40 arquivos em:

```
results/layer_<N>/<categoria>/<label>/<variant>/stringent/recognition/recognition_1..10.txt
results/layer_<N>/<categoria>/<label>/<variant>/stringent/eval/eval_1..10.txt
results/layer_<N>/<categoria>/<label>/<variant>/lenient/recognition/recognition_1..10.txt
results/layer_<N>/<categoria>/<label>/<variant>/lenient/eval/eval_1..10.txt
```

- `recognition_N.txt` = o que o modelo respondeu (texto livre, geralmente uma palavra).
- `eval_N.txt` = `0` ou `1` — se um segundo modelo julgou que a resposta menciona o conceito esperado.
- `stringent` = pergunta neutra ("What is in the image?"); `lenient` = pergunta guiada pela categoria ("What Animals is in the image if you had to guess?").
- 10 rodadas de cada pra suavizar a variabilidade do modelo.

## 4. Ver os resultados

### Baixando direto do S3

```bash
# um arquivo
aws s3 cp s3://steering-visualization/results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt - \
  --profile steering-visualization

# a pasta inteira de uma imagem
aws s3 sync s3://steering-visualization/results/layer_0/Animals/Giraffe/0/ ./giraffe-0-results/ \
  --profile steering-visualization
```

### Consultando via Athena (recomendado pra análise)

Abra o console do Athena (workgroup **`steering-visualization`**, banco
**`steering_visualization`**) —
veja https://docs.aws.amazon.com/athena/latest/ug/querying-athena-tables.html
se for a primeira vez usando o editor de query do Athena.

Duas tabelas: `recognition_results` (coluna `output`) e `evaluation_results`
(coluna `evaluation`, `"0"` ou `"1"`). Ambas particionadas por `layer`,
`category`, `label`, `variant`, `prompt_type`.

**Ver as respostas cruas de uma imagem:**
```sql
SELECT variant, prompt_type, output
FROM steering_visualization.recognition_results
WHERE layer = '0' AND category = 'Animals' AND label = 'Giraffe'
ORDER BY variant, prompt_type;
```

**Taxa de sucesso por imagem-exemplo (variant), pra comparar imagens diferentes do mesmo conceito:**
```sql
SELECT variant, prompt_type, count(*) AS rodadas, sum(cast(evaluation AS integer)) AS sucessos
FROM steering_visualization.evaluation_results
WHERE layer = '0' AND label = 'Giraffe'
GROUP BY variant, prompt_type
ORDER BY variant, prompt_type;
```

**Taxa de sucesso agregada por conceito, em todas as layers:**
```sql
SELECT category, label, prompt_type, avg(cast(evaluation AS double)) AS taxa_sucesso
FROM steering_visualization.evaluation_results
GROUP BY category, label, prompt_type
ORDER BY category, label, prompt_type;
```

**Extrair o número da rodada e juntar recognition + eval do mesmo round** (o
número da rodada não é uma coluna direta — vem do nome do arquivo via a
pseudo-coluna `"$path"`):
```sql
SELECT category, label, variant, prompt_type,
       regexp_extract("$path", 'recognition_(\d+)\.txt', 1) AS round,
       output
FROM steering_visualization.recognition_results
WHERE layer = '0' AND label = 'Giraffe';
```

## Links úteis

- Trocar de role no console: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-console.html
- Configurar profile do CLI que assume role automaticamente: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html
- Rodar queries no editor do Athena: https://docs.aws.amazon.com/athena/latest/ug/querying-athena-tables.html
- Referência de comandos S3 no CLI: https://docs.aws.amazon.com/cli/latest/reference/s3/
