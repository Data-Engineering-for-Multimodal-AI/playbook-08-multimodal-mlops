# Chunked MLOps Pipeline Bundle

The `mlops_pipeline.bundle` file has been split into smaller parts to comply with GitHub's file size limits.

## Reassembling the Bundle

To reassemble the original bundle file, use one of the following methods:

### On Unix/Linux/macOS

```bash
cat part_* > ../reassembled_mlops_pipeline.bundle
```

### On Windows

```powershell
copy /b part_aa+part_ab+part_ac ..\reassembled_mlops_pipeline.bundle
```

After reassembling, the file should be 178MB and identical to the original `mlops_pipeline.bundle`. 