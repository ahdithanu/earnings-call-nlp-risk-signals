# Deploying the scoring API on AWS

The API runs on **AWS App Runner** (managed containers: TLS, scaling, and
health checks included), pulled from a private **ECR** repository, deployed
by **GitHub Actions over OIDC** — no AWS access keys are ever created or
stored anywhere. Everything is code in this repo; the AWS console is only
used once, to apply the bootstrap stack.

```
GitHub Actions ──OIDC──▶ DeployRole ──▶ push image to ECR
                                   └──▶ create App Runner service (first run)
ECR :latest push ─────────────────────▶ App Runner auto-deploys thereafter
```

## One-time bootstrap (~5 minutes, all in the browser)

1. **AWS console → CloudFormation → Create stack → Upload a template file**
   and pick [`infra/aws/bootstrap.yaml`](../infra/aws/bootstrap.yaml).
   Choose your region first (top-right — e.g. `us-east-1`) and remember it.
2. Stack name: `earnings-signals-bootstrap`. The `GitHubRepository`
   parameter already defaults to this repo. Create the stack (acknowledge
   the IAM-resources checkbox — it creates the two roles described above).
3. When the stack is `CREATE_COMPLETE`, open its **Outputs** tab and copy
   `DeployRoleArn`.
4. **GitHub → repo → Settings → Secrets and variables → Actions →
   Variables** (variables, not secrets — neither value is sensitive):
   - `AWS_DEPLOY_ROLE_ARN` = the copied ARN
   - `AWS_REGION` = the region from step 1

## Deploying

- **Actions → "Deploy to AWS" → Run workflow** (or push a `v*` tag — a
  GitHub Release does this). The first run creates the App Runner service
  and prints its URL in the run summary; later runs just push `:latest`,
  which App Runner auto-deploys.
- Verify: `https://<service-url>/healthz`, interactive docs at `/docs`.

## Wiring the site's live scorer

Put the service origin (e.g. `https://xyz.us-east-1.awsapprunner.com`) into
the `signals-api` meta tag in `web/index.template.html`, run
`make site`, and commit — the explorer's "score your own text" section
activates automatically once `/healthz` answers.

## Cost and teardown

The service runs one 0.25 vCPU / 0.5 GB instance: roughly **$2–5/month**
idle (App Runner bills provisioned memory while idle, vCPU only under
traffic), well within typical portfolio budgets; ECR storage is pennies.
To tear everything down: delete the App Runner service (console or
`aws apprunner delete-service`), then delete the CloudFormation stack
(which removes the ECR repo, roles, and OIDC provider).

## Notes

- The deploy role's trust policy is pinned to `repo:ahdithanu/earnings-call-nlp-risk-signals:*`
  — no other repository (or fork) can assume it.
- Every IAM statement is resource-scoped except App Runner management,
  whose service ARNs don't exist until first deploy (noted inline in the
  template).
- If your AWS account already has a GitHub OIDC provider (one per account),
  delete the `GitHubOidcProvider` resource from the template and point
  `DeployRole`'s `Federated` principal at the existing provider's ARN.
