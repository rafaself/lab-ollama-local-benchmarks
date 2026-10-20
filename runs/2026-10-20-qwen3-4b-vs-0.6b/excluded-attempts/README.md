# Excluded validation attempts

These raw files preserve the two completed validation attempts before the recorded comparison. They are excluded from the final performance tables and quality comparison.

1. `01-api-thinking-false.json`: the request set `think: false`. The installed 4B tag still emitted reasoning into the response, and the 256-token quality budget often ended before the final answer. A quality score from this setup would confuse a truncated reasoning trace with a failed completed answer. One 0.6B short request also ended below the output cap.
2. `02-native-no-thinking-prefix.json`: an attempt to standardize non-thinking behavior using a raw Qwen ChatML prompt with an empty thinking prefix and `/no_think`. The 4B responses still contained reasoning. One 0.6B short request also ended below the output cap. This is not the protocol used in the final comparison.

The final comparison explicitly used `think: true` for both models, native templates, and up to 2,048 output tokens on quality questions. Its data are in `../results.json`. The exact standalone scripts for these two preliminary attempts were not archived; their JSON retains the workloads, request settings, returned text, measurements, and available model metadata. These files document exclusion decisions and must not be pooled into the final comparison.
