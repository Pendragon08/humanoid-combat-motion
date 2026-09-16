# ReMoCap execution status

Data acquisition, recording-preserving conversion, independent GPU model integration, 200-step pilots and four-arm validation smoke are complete.

Main training is running: ReMoS-body-adapted on GPU 0 and InterGen-body-adapted on GPU 1, each 3 seeds x 5000 updates. Evaluation queues wait for their method's training queue to complete, then run raw generation on the entire test split and four refinement arms on one fixed window per test recording.

The finite local archive process will copy aggregate results and run logs, then commit and push only this run's records when the server queues complete or fail. It excludes licensed motions, third-party source and model weights. Its status is recorded in archive_watcher_status.json; final Git publication receipt is local under .aris/remocap_archive_publication.json.

This is an adapted body-only comparison and component ablation. It is not original-paper reproduction, independently verified combat contact, the full proposed algorithm, or downstream robot learning validation. Those limits remain explicit in the protocol.
