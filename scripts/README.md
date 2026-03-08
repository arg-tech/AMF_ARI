# Model Export Scripts

The app loads a pre-compiled OpenVINO model from HuggingFace (`ov_model_path` in `config/config.json`).

## Re-exporting the model

If the source model changes or you need to re-export:

```bash
# Build the export container
docker build --platform linux/amd64 -f scripts/Dockerfile -t amf-ari-export .

# Export locally
docker run --rm -v "$(pwd)/exported_model:/app/exported_model" amf-ari-export

# Push to HuggingFace Hub
docker run --rm \
  -v "$(pwd)/exported_model:/app/exported_model" \
  -e HUGGING_FACE_HUB_TOKEN=<your-token> \
  amf-ari-export python scripts/export_model.py --push-to-hub arg-tech/amf-ari-roberta-ov-int8
```

Then update `ov_model_path` in `config/config.json` if the repo ID changed.
