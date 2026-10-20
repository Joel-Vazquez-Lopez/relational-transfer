# Prospective restart and resource amendment

The prior attempt at commit5dce684 completed only one exposed reference forward, then stopped at its fixed absolute2GiB swap cap. No donor recapture or intervention outcome exists from that attempt. Its directory and STARTED/ERROR records remain untouched. The user's subsequent instruction authorizes another independent implementation and necessary changes. This is a distinct new attempt, not a silent retry.

Read-only host assessment found normal memory-pressure level1 and55% system-wide memory free according to `memory_pressure -Q`, while encrypted swap remained around2.08GiB. Allocated swap by itself does not establish current critical memory pressure or a model crash. The initial absolute cap is therefore replaced prospectively with a multi-indicator guard, not removed:

- Require `kern.memorystatus_vm_pressure_level == 1` before dispatch, every5s during a forward and before every worker load; warning/critical stops the run.
- Disk reserve increased from4GiB to8GiB.
- Absolute swap cap4GiB AND swap growth strictly less than1GiB from the first pre-load resource measurement of this attempt. The initial value is logged on every dispatch.
- Existing6GiB MLX memory guideline/active allocation check,256MiB cache limit,32 actual forwards per worker,90s per forward and45-minute dispatch-start window remain.
- Every attempted forward counts, including exposed restart checks. No automatic retry or continuation after a failed technical/resource gate. No unrelated user app is quit, no process is killed for headroom, and no unique file is deleted.

Normal/warning/critical memory-pressure constants1/2/4 are documented in the installed Apple SDK's `dispatch/source.h` (lines235–257). The raw kernel flag is used only as a conservative operational signal, not a scientific variable. These limits reduce risk but cannot guarantee uninterrupted host operation.

Scientific source/prompt/native scoring and literal rival predictions are inherited. The only scientific extension is the predeclared7/14/21 depth profile, described in the rationale and protocol before outcomes. This amendment does not repair, relabel or erase the old abort, or relax any numerical equivalence tolerance.
