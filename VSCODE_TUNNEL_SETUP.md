# VS Code Remote Tunnel setup

## Allocation and CLI

- Compute hostname: `evc1`
- Slurm job ID: `689089`
- Slurm nodelist: `evc1`
- Architecture: `x86_64`
- VS Code CLI: `/home/ad906660/.local/bin/code`
- Tunnel name: `newton-omnitom`
- Log path: `~/omnitom_da_experiment/logs/vscode_tunnel.log`

This was configured from a Slurm compute allocation, not a login node. The
standalone VS Code CLI was installed without sudo because the pre-existing
server-side CLI wrapper did not respond to CLI help/version commands.

## Tunnel commands

The tunnel runs in the `vscode-tunnel` tmux session:

```bash
export PATH="$HOME/.local/bin:$PATH"
tmux new-session -d -s vscode-tunnel \
  'export PATH="$HOME/.local/bin:$PATH"; cd "$HOME/omnitom_da_experiment"; code tunnel --accept-server-license-terms --name newton-omnitom 2>&1 | tee -a "$HOME/omnitom_da_experiment/logs/vscode_tunnel.log"'
```

If the initial run prompts for an account provider through the piped tmux
command, first perform the manual login in that same session, then start the
tunnel command above after browser authentication completes:

```bash
tmux new-session -d -s vscode-tunnel \
  'export PATH="$HOME/.local/bin:$PATH"; cd "$HOME/omnitom_da_experiment"; code tunnel user login --provider microsoft 2>&1 | tee -a "$HOME/omnitom_da_experiment/logs/vscode_tunnel.log"'
```

At setup time, this manual device-login step was awaiting browser
authentication. The one-time device code is intentionally not recorded in this
file.

Inspect the session:

```bash
tmux attach -t vscode-tunnel
```

Detach with `Ctrl-b d`.

View logs:

```bash
tail -f ~/omnitom_da_experiment/logs/vscode_tunnel.log
```

Stop the tunnel:

```bash
tmux kill-session -t vscode-tunnel
```

The tunnel stops when the Slurm allocation ends. On a new compute node, rerun
the same tunnel command with the same tunnel name. No SSH config update is
required when the compute hostname changes.

## Connect from macOS VS Code

1. Install **Remote - Tunnels**.
2. Open the Command Palette.
3. Run `Remote Tunnels: Connect to Tunnel...`.
4. Select `newton-omnitom`.
5. Open `~/omnitom_da_experiment`.
