# m4_remote — running the TAReL Docker demo headless from an iPhone

Recipe for running `docker compose up --build` on an M-series MacBook Pro
with the lid closed, then watching battery, logs, and generated files
from an iPhone over LTE/Wi-Fi. Stock macOS + free apps; no port-forwarding,
no public IP.

## Setup (once)

1. **Tailscale** — install on the Mac (`brew install tailscale && sudo
   tailscale up --ssh`) and from the App Store on the iPhone, sign in
   with the same account on both. `--ssh` lets Tailscale handle SSH
   auth so you don't ship keys to the phone.
2. **iPhone SSH client** — Blink Shell (paid, polished) or Termius
   (free tier is enough). Add the Mac's Tailscale name as a host; auth
   is automatic via Tailscale SSH.
3. **Lid-closed-awake** — either Amphetamine from the Mac App Store
   (free; one-click "close lid" override) or one-shot:
   ```bash
   sudo pmset -c disablesleep 1     # AC only; restore with 0
   ```
4. **ntfy** (optional, push alerts) — install ntfy from the iOS App
   Store, pick a private topic name (any random string), subscribe to it.
   No account needed.

Keep the Mac plugged in. Even "awake with lid closed" drains the battery
faster than you'd expect because fans ramp under cargo + torch load.

## Launching the demo

SSH in from the iPhone, then start the build inside `tmux` so a network
blip doesn't kill it. `caffeinate` is belt-and-braces on top of the
pmset / Amphetamine sleep override.

```bash
cd ~/git/research/tarel
caffeinate -dis tmux new -d -s tarel \
    "docker compose up --build &> ~/tarel-demo.log; \
     curl -d 'tarel: compose exited' ntfy.sh/<your-topic>"
```

Detach safely (`Ctrl-b d`) or just close the SSH session — `tmux` keeps
the job alive. Reattach later with `tmux attach -t tarel`.

## Monitoring from the iPhone

All over the same SSH session.

| What                | Command                                                   |
| ------------------- | --------------------------------------------------------- |
| Live build logs     | `tail -f ~/tarel-demo.log`                                |
| Container state     | `docker compose ps`                                       |
| Per-service logs    | `docker compose logs -f stage-one`                        |
| Generated files     | `ls -lh ~/git/research/tarel/artifacts/runs/`             |
| Latest run dir      | `ls -1dt ~/git/research/tarel/artifacts/runs/*/ \| head`  |
| Battery %, charging | `pmset -g batt \| head -2`                                |
| CPU / memory load   | `top -l 1 -n 0` or `htop` (after `brew install htop`)     |
| Disk left           | `df -h /`                                                 |

To watch a value continuously: `watch -n 30 'pmset -g batt | head -2'`
(install with `brew install watch`).

## Push notifications without polling

A two-line shell loop turns the Mac into a publisher and the iPhone
ntfy app into the subscriber. Drop this in the same `tmux` session
under a second window (`Ctrl-b c`):

```bash
# Battery dip alert
while sleep 600; do
  PCT=$(pmset -g batt | awk -F'[%\t]' '/InternalBattery/ {print $2}')
  AC=$(pmset -g batt | awk '/Now drawing/ {print $4}')
  [[ "$AC" == "'Battery" && "$PCT" -lt 30 ]] && \
      curl -d "tarel: $PCT% on battery" ntfy.sh/<your-topic>
done &

# New run-dir alert
fswatch -0 ~/git/research/tarel/artifacts/runs | while IFS= read -r -d '' f; do
    curl -d "tarel: artifact $(basename "$f")" ntfy.sh/<your-topic>
done &
```

`brew install fswatch` if needed.

## Viewing generated files on the phone

Two options, pick whichever fits the moment:

- **Inline via SSH** — for parquet / log / small JSON files, `cat`,
  `less`, `jq` on the phone shell are fine. For PNG sanity grids, scp
  to the iPhone via the SSH client's file manager (Blink and Termius
  both have one) and tap to preview.
- **Auto-sync to iCloud Drive** — symlink the runs directory under
  iCloud so the Files app on the iPhone sees every output as it lands:
  ```bash
  ln -s ~/git/research/tarel/artifacts/runs \
        "$HOME/Library/Mobile Documents/com~apple~CloudDocs/tarel-runs"
  ```
  Slower to propagate than SSH but works offline and renders PNGs
  natively. Note iCloud will refuse to sync symlink targets larger than
  ~50 GB; the demo run is well under that.

## Full screen mirroring (optional)

If you want the actual Mac desktop (e.g. to look at a Docker Desktop
GUI graph), install Screens 5 or Jump Desktop on the iPhone — both
discover Macs over Tailscale automatically once VNC / Screen Sharing
is enabled in System Settings → General → Sharing. Overkill for a
batch job; useful for ad-hoc inspection.

## Teardown

```bash
tmux kill-session -t tarel
sudo pmset -c disablesleep 0     # if you used the pmset route
```

Amphetamine and Tailscale can stay enabled — they cost nothing when idle.
