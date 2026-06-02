# Public Multi-Generator Availability Audit

- targetdiff_sampling_results_google_drive: blocked_or_unreachable; rc=28; elapsed=10.23s
  URL: https://drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS?usp=share_link
- targetdiff_repo: reachable; rc=0; elapsed=1.21s
  URL: https://github.com/guanjq/targetdiff
- pocket2mol_repo: blocked_or_unreachable; rc=28; elapsed=20.01s
  URL: https://github.com/pengxingang/Pocket2Mol
- decompdiff_repo: blocked_or_unreachable; rc=28; elapsed=20.01s
  URL: https://github.com/bytedance/DecompDiff

TargetDiff official README states that docked meta files are provided for TargetDiff, CVAE, AR, and Pocket2Mol via the sampling-results Google Drive folder. If the folder cannot be reached from the experiment server, these generator-generalization experiments need local/uploaded meta files rather than being mixed with DiffSBDD-only evidence.

## Follow-Up

The TargetDiff Google Drive folder was retried with `gdown --folder` from the experiment server. It failed before file enumeration:

`HTTPSConnectionPool(host='drive.google.com', port=443): Failed to establish a new connection: [Errno 101] Network is unreachable`

The same run successfully downloaded the standard 100-pocket CrossDocked test receptor archive from the HuggingFace mirror `hf-mirror.com`, which confirms that the failure is specific to the public generated-sample hosting route rather than a general lack of network. Therefore the multi-generator experiment should be resumed when TargetDiff/Pocket2Mol/DecompDiff generated SDF/PT files are uploaded locally or mirrored to an accessible source.
