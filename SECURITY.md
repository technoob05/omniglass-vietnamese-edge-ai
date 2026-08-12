# Security and privacy

- Bind development services to `127.0.0.1` and use an authenticated SSH tunnel.
- Do not expose the research gateway directly to the public internet.
- Never commit API tokens, certificates, model weights, raw camera/audio data,
  session traces, or generated evidence containing people.
- The Vietnamese profile captures one current camera frame only after a
  finalized utterance; raw media logging is disabled by design.
- Report security issues privately to the repository owner instead of opening
  an issue containing credentials, recordings, or infrastructure addresses.

This is a research visual-assistance prototype, not a certified navigation or
hazard-avoidance system.
