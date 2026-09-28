# Convidando colegas para o pipeline de avaliação (guia do admin)

*(Tradução em inglês: [`admin-invite-collaborators.en.md`](admin-invite-collaborators.en.md))*

Este guia é para você (o dono da conta AWS). Ele mostra como dar a um colega
acesso para subir imagens, consultar o Athena e baixar resultados — sem dar
acesso a mais nada na conta.

O CDK já criou um **role** (`steering-visualization-collaborator`) com
exatamente as permissões necessárias. Ele não pode ser assumido por ninguém
até você explicitamente autorizar cada usuário — isso é feito uma vez por
colega, fora do CDK (de propósito: a lista de quem tem acesso não devia
exigir redeploy de infraestrutura toda vez que alguém entra ou sai do time).

Descubra o Account ID da sua conta de deploy (não gravamos esse número em
nenhum arquivo do repositório, de propósito — ver seção 12 da spec):
```bash
aws sts get-caller-identity --profile <seu-profile> --query Account --output text
```
Nos comandos abaixo, `<ACCOUNT_ID>` é esse número.

- Role ARN: `arn:aws:iam::<ACCOUNT_ID>:role/steering-visualization-collaborator`
- Sessão dura até 12h (o máximo permitido) — colega não precisa reassumir o role toda hora.

## Passo 0 (uma vez só): criar o grupo que autoriza assumir o role

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

Qualquer usuário IAM colocado nesse grupo poderá assumir o role. Você só
precisa fazer isso uma vez; os próximos passos são por colega.

## Passo 1 (por colega): criar o usuário IAM

```bash
aws iam create-user --user-name alice
aws iam add-user-to-group --group-name steering-visualization-collaborators --user-name alice
```

## Passo 2: dar uma forma de autenticação a ela

Escolha uma (ou as duas):

**Acesso ao console AWS** (bom se ela vai usar Athena/S3 pela interface web):
```bash
aws iam create-login-profile \
  --user-name alice \
  --password 'TrocarEssaSenha123!' \
  --password-reset-required
```
Passe pra ela: o **Account ID** (o número que você pegou acima), o usuário
(`alice`) e essa senha temporária — ela troca no primeiro login em
`https://<ACCOUNT_ID>.signin.aws.amazon.com/console`.

**Access key para linha de comando** (bom se ela vai usar o AWS CLI):
```bash
aws iam create-access-key --user-name alice
```
Isso imprime `AccessKeyId` e `SecretAccessKey` — mande pra ela por um canal
seguro (não por e-mail/Slack em texto puro). Essa é a única vez que a
`SecretAccessKey` é mostrada.

## Passo 3: mandar pra ela

- O **guia do colega**: `evaluation/docs/collaborator-guide.md`
- O **Account ID**: o número que você pegou no início deste guia
- O **nome do role**: `steering-visualization-collaborator`
- As credenciais do Passo 2

## Removendo o acesso de alguém

```bash
aws iam remove-user-from-group --group-name steering-visualization-collaborators --user-name alice
# opcional: apagar o usuário de vez
aws iam delete-login-profile --user-name alice          # se criou senha de console
aws iam list-access-keys --user-name alice               # pra achar o access key id
aws iam delete-access-key --user-name alice --access-key-id <ID>
aws iam delete-user --user-name alice
```

Remover do grupo já é suficiente para revogar o acesso ao role — não precisa
apagar o usuário se for só uma pausa temporária.
