# Same-implementation replay

Run the original joint model for the same four settings using the diagnostic
script's exact execution settings (including two CPU threads). This avoids
attributing a change to padding when a numerical or implementation difference
between the original training runner and diagnostic runner could contribute.
Compare padding primarily to these matched replays; also report agreement
with the archived originals. The original variant adds no model or data
change and uses the same initialization, loss, optimizer, and episode schedule.
