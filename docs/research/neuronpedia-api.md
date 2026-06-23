# Neuronpedia API — reference for the gemma-4-E2B positive control

Neuronpedia hosts the **Gemma 4 SAEs** release (David Chanin / Decode Research, Jun 2026):
model id `gemma-4-e2b`, source/SAE `17-matryoshka-res-65k` (residual stream, layer 17,
65k features). We use its API as a third-party-attested positive control for E1: fetch a
documented feature's top-activating tokens, run the same text through our local capture,
and check the local activation at layer 17 matches what Neuronpedia reports.

> The public API is marked **WIP** on the docs. The GET data endpoints below are stable;
> for POST/inference bodies, confirm the exact shape in the live interactive spec
> (https://neuronpedia.org/api-doc) or prefer the official Python client, which tracks the
> API and won't break on minor changes.

## Identifiers we use

| Field | Value |
|---|---|
| `modelId` | `gemma-4-e2b` |
| `source` (a.k.a. `layer`) | `17-matryoshka-res-65k` |
| `index` | the feature number, e.g. `12345` |

## Two distinct APIs

1. **Data API** (`https://www.neuronpedia.org/api/...`) — read precomputed feature data
   (explanations, top-activating snippets + per-token values, top positive/negative logit
   tokens). HTTP only, no GPU. **This is what the positive control needs.**
2. **Inference API** (`neuronpedia_inference_client`, separate PyPI pkg) — runs model+SAE
   server-side to compute activations on arbitrary text or to steer. Heavier; needs their
   hosted inference (or self-host). Optional for us.

## Auth

Read endpoints are largely public. Write/account/inference endpoints need an API key:
get it at https://neuronpedia.org/account, send it as header `x-api-key: <KEY>`.
Per project rule, read the key from the OS env (`NEURONPEDIA_API_KEY`) via `config.py` —
never hardcode. For bulk, prefer the S3 exports over hammering the API.

## Endpoint 1 — GET a feature (stable)

```
GET https://www.neuronpedia.org/api/feature/{modelId}/{source}/{index}
```

Example (curl):

```bash
curl -s https://www.neuronpedia.org/api/feature/gemma-4-e2b/17-matryoshka-res-65k/12345 \
  | python3 -m json.tool
```

Returns (shape, abridged): the feature's `explanations[]` (auto-interp labels),
`activations[]` (top-activating text snippets with per-token `values[]`), and the top
positive/negative `logits`/tokens the feature writes to. The per-token `values` and the
top tokens are exactly what we compare our local capture against.

## Endpoint 2 — activation of a feature on YOUR text (verify body in live spec)

To get a feature's activation per token on custom text (the cleanest apparatus check):

```
POST https://www.neuronpedia.org/api/activation/new
Content-Type: application/json
x-api-key: <KEY>           # if required

# body shape — CONFIRM in https://neuronpedia.org/api-doc; approximately:
{ "feature": { "modelId": "gemma-4-e2b",
               "source": "17-matryoshka-res-65k",
               "index": 12345 },
  "customText": "The capital of France is Paris" }
```

Returns per-token activation values of that feature on your text. (If the body differs in
the live spec, use the Python client below, which always matches the server.)

## Endpoint 3 — search features by explanation (find a target feature)

```
# Confirm exact path/verb in the live spec; conceptually:
GET/POST https://www.neuronpedia.org/api/explanation/search?modelId=gemma-4-e2b&query=capital city
```

Use this to locate, e.g., an entity / factual-recall feature to use as the control.

## Official Python client (recommended — survives API changes)

```bash
pip install neuronpedia            # data API client
# pip install neuronpedia_inference_client   # only if using server-side inference/steering
```

```python
# Pattern (confirm against the installed version's docstrings):
from neuronpedia.feature import Feature
f = Feature.get(model_id="gemma-4-e2b",
                source="17-matryoshka-res-65k",
                index=12345)
print(f.explanations)          # auto-interp labels
print(f.activations[:3])       # top-activating snippets + per-token values
# top logit tokens the feature promotes -> compare to our unembedding readout
```

## Bulk exports (avoid hammering the API)

Full precomputed data dumps (features, activations, explanations) live in S3:

```
https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/
```

For the whole `gemma-4-e2b / 17-matryoshka-res-65k` SAE, download the export once and read
locally — better than thousands of API calls.

## How this becomes the E1 positive control

1. Pick (or search) a feature with a clear, factual/entity explanation on
   `gemma-4-e2b/17-matryoshka-res-65k`.
2. Fetch its top-activating tokens (Endpoint 1) and, ideally, its activation on a few of
   our own E1 prompts (Endpoint 2).
3. Run those exact prompts through our local `ActivationRecorder` capturing **layer 17**,
   then project the residual onto the SAE encoder row for that feature
   (`W_enc[index]`, from the downloaded SAE weights / SAELens).
4. **Attested check:** our local per-token activation should track Neuronpedia's reported
   values (high correlation, same top-activating positions). If it does → our apparatus
   (layer indexing, chat template, residual readout) is validated against a third party on
   the real Gemma 4 E2B. If it doesn't → we've localized exactly which Gemma-4 detail we're
   breaking. This is a non-tautological positive control at the feature/logit level — the
   thing `d_unembed` (tautological) and `d_refusal` (behavioral, failed) didn't give.

## Notes / caveats

- API is WIP; treat POST/search bodies as "confirm live or use the client".
- The SAE is on **layer 17**; our E1 `d_know` extraction layer is 26. The control validates
  the apparatus at layer 17; if we want the control at 26 we need an SAE trained there (the
  release may include other layers — check the source list on the model page).
- SAE weights themselves (for `W_enc`) load via **SAELens** (`pip install sae-lens`) using
  the same release id — confirm the release is registered in SAELens' pretrained directory.
