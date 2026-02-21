# Agent setup runbook — ECM Operations on your local machine

Step-by-step runbook for ops/agents to set up the ECM Operations plugin on their laptop so they can run **"run ECM"**, **"order &lt;id&gt;"**, **"my tickets"**, etc. via Claude (Cursor or Claude Code).

---

## Prerequisites

| Requirement | Check |
|-------------|--------|
| **Claude** | Cursor (with Claude) or Claude Code installed and working. |
| **Repo access** | You can clone or open the repo that contains `work-plugins/ecm-operations` (e.g. `ai-velocity`). |
| **VPN (if using ECS gateway)** | You must be on **corporate VPN** (or inside the VPC) to reach the internal ECS gateway. |
| **Docker (only if using local gateway)** | Required only if you choose **Option B: Local Docker** below. |

---

## Step 1: Open the ECM Operations workspace

1. Clone or open the repo that contains the plugin, e.g.:
   ```bash
   git clone <your-org>/ai-velocity.git
   cd ai-velocity
   ```
2. In **Cursor** or **Claude Code**, open the folder that contains the ECM plugin. Either:
   - Open the **repo root** (e.g. `ai-velocity`), or  
   - Open **`work-plugins/ecm-operations`** directly.
3. Confirm you see: `AGENTS.md`, `queries/`, `skills/`, `config/`, `.mcp.json` inside the opened folder (or under `work-plugins/ecm-operations`).

---

## Step 2: Choose how you connect to the gateway

You connect to **one** MCP gateway that provides Redshift + Google Sheets. Pick one:

| Option | When to use | You need |
|--------|-------------|----------|
| **A: ECS gateway** | You have VPN and want zero local setup. | VPN on; no Docker. |
| **B: Local Docker gateway** | No VPN, or you develop/debug the gateway. | Docker installed; gateway repo/scripts. |

---

### Option A: ECS gateway (recommended if you have VPN)

1. Stay on **VPN** (required to reach the internal gateway).
2. The plugin’s **`.mcp.json`** should point to the ECS gateway. Example:
   ```json
   {
     "mcpServers": {
       "ecm-gateway": {
         "type": "http",
         "url": "http://internal-ecm-gateway-svc-464678860.eu-west-2.elb.amazonaws.com:80/mcp"
       }
     }
   }
   ```
3. If your repo already has this in `work-plugins/ecm-operations/.mcp.json`, **do nothing** here.  
   If you use a **different** MCP config (e.g. Cursor user settings), add the same `ecm-gateway` entry with the URL above (port **80**, path **/mcp**).
4. Go to **Step 3**.

---

### Option B: Local Docker gateway

1. Install **Docker** and ensure it’s running.
2. **Agent Gateway (path-based routing)** — Go binary + nginx on 8080:
     ```bash
     cd work-plugins/ecm-operations/_backup/gateway
     cp .env.example .env   # fill Redshift + Sheets credentials
     docker compose up -d
     ```
     You should see **mcp-router** (nginx on 8080) and **agent-gateway** (image `ghcr.io/agentgateway/agentgateway`).
   - **Python gateway** — FastMCP proxy in `_backup/gateway/`; runs as `python app.py`. Use only if you need the Python stack (e.g. audit DB, custom middleware). Run from `_backup/gateway/` with that repo’s `docker-compose.yml`.
3. Point MCP at **localhost** — unified: `http://localhost:8080/mcp`; or path-based (fault isolation): `http://localhost:8080/mcp/redshift`, `http://localhost:8080/mcp/sheets`.
   ```json
   {
     "mcpServers": {
       "ecm-gateway": {
         "type": "http",
         "url": "http://localhost:8080/mcp"
       }
     }
   }
   ```
4. Put this in `work-plugins/ecm-operations/.mcp.json` **or** in your Cursor/Claude MCP config so the **ecm-gateway** server uses `http://localhost:8080/mcp`.
5. Go to **Step 3**.

---

## Step 3: Ensure MCP uses this project’s config

- **Cursor**: MCP can be configured **per project**. If there is a **`.mcp.json`** in `work-plugins/ecm-operations/`, Cursor may load it when that folder (or the repo) is the workspace. Confirm in **Settings → MCP** that **ecm-gateway** appears and uses the URL you chose (ECS or localhost).
- **Claude Code**: Use the MCP config that includes the **ecm-gateway** entry (same URL as above). If the docs say “add to your MCP config”, merge the `ecm-gateway` block into that file.

Do **not** add a separate Redshift MCP (e.g. `awslabs.redshift-mcp-server`). All Redshift and Sheets access goes through **ecm-gateway** only.

---

## Checking what MCP tools are configured (Claude Code / Cursor)

If **ecm-gateway** is “not available” or tools are missing, the IDE may be loading a different MCP config.

| Where | File | Contents |
|-------|------|----------|
| **Repo root** | `ai-velocity/.mcp.json` | Example/placeholder only — **no ecm-gateway**. |
| **Plugin** | `work-plugins/ecm-operations/.mcp.json` | **ecm-gateway** with ECS (or localhost) URL. |

- **Cursor**: **Settings → MCP** shows which servers are loaded. If you opened the **repo root**, Cursor may use the root `.mcp.json` (no ecm-gateway). Fix: add the `ecm-gateway` block from `work-plugins/ecm-operations/.mcp.json` into the root `.mcp.json`, or open the folder `work-plugins/ecm-operations` as the workspace so the plugin’s `.mcp.json` is used.
- **Claude Code**: MCP is configured by **scope** (project `.mcp.json`, user config, or managed). Use **Settings** (e.g. `/config` or the settings UI) and check which MCP servers are listed. If **ecm-gateway** is missing, the config currently in use doesn’t include it. Fix: merge the `ecm-gateway` entry from `work-plugins/ecm-operations/.mcp.json` into the **same config file your IDE uses** (e.g. the repo root `.mcp.json` if that’s the project config, or your user-level MCP config).

**Quick fix when opening the repo root:** Copy the `ecm-gateway` server from `work-plugins/ecm-operations/.mcp.json` into `ai-velocity/.mcp.json` (under `mcpServers`), then reload the window so ecm-gateway is available.

---

## Step 4: Trust the plugin (if prompted)

When you first use the plugin, you may see: *"Make sure you trust a plugin before installing or using it..."*

- This plugin is **internal** (Aspora ECM). You can **trust it** so that Claude can read `AGENTS.md`, skills, and queries and call the ecm-gateway MCP tools.
- Approve/trust when prompted so the plugin can run.

---

## Step 5: Reload MCP and window (important)

After changing `.mcp.json` or MCP settings:

1. **Reload the window**: Command Palette → **“Developer: Reload Window”** (or restart Cursor/Claude Code).
2. Re-open the same workspace so MCP reconnects to **ecm-gateway** with the new URL.

---

## Step 6: Verify setup

1. In the chat, ask Claude to run ECM, for example:
   - **"Run ECM"** or **"Show ECM dashboard"**
   - **"order AE13ATCYOY00"** (or any real order id)
2. Claude should:
   - Use **ecm-gateway** MCP (no other Redshift MCP).
   - Run `queries/ecm-pending-list.sql` for the dashboard and `queries/ecm-order-detail-v2.sql` for order detail.
   - Return real data (or a clear error if gateway/network fails).

If you see **"Tool not found"** or **"ecm-gateway not available"**, MCP did not load the ecm-gateway server — repeat Step 3 and Step 5.

---

## Troubleshooting

| Symptom | What to do |
|--------|------------|
| **"localhost:8080/mcp is not responding"** | Your IDE MCP config is using the **local** gateway URL but nothing is listening. **Fix A (use ECS):** In **Settings → MCP**, set ecm-gateway URL to `http://internal-ecm-gateway-svc-464678860.eu-west-2.elb.amazonaws.com:80/mcp` and stay on **VPN**. **Fix B (use local):** Start the gateway (e.g. `docker compose up -d` in the ecm-gateway repo). Then reload the window. |
| **"Tool not found"** / **ecm-gateway not available** | MCP not loaded. Confirm `.mcp.json` (or your MCP config) has `ecm-gateway` with the correct URL. Reload window (Step 5). |
| **"Request timed out"** / **MCP error -32001** | Gateway not reachable. **ECS**: Are you on VPN? **Local**: Is `docker compose up` running and healthy? |
| **Empty or no data** | Gateway may be up but backend (Redshift/Sheets) failing. Check gateway logs (local: `docker logs <gateway-container>`; ECS: CloudWatch). |
| **Wrong gateway URL** | ECS uses port **80** and path **/mcp**. Local uses **localhost:8080** and path **/mcp**. No trailing slash. |
| **Plugin not used** | Open the workspace that contains `work-plugins/ecm-operations` (or the repo root). Claude must see `AGENTS.md` and the plugin files. |
| **localhost:8080/mcp works, ECS URL doesn’t** | Local Docker (Agent Gateway) is fine; the **ECS** gateway at `internal-ecm-gateway-svc-...` is failing. See **“ECS vs local”** below. |

### ECS vs local — why ECS might not work when local does

If **localhost:8080/mcp** works (local Docker with Agent Gateway) but **internal-ecm-gateway-svc-464678860.eu-west-2.elb.amazonaws.com:80/mcp** does not:

| Check | What to do |
|-------|------------|
| **1. Reachability** | You must be on **corporate VPN** (or inside the VPC) to reach the internal ALB. Try `curl -v http://internal-ecm-gateway-svc-464678860.eu-west-2.elb.amazonaws.com:80/mcp` — connection refused or timeout = network/VPN. |
| **2. 404 from gateway / no request logs in sheets-mcp** | ECS gateway logs show `upstream error: 404 Not Found` → one **upstream** is failing. If **sheets-mcp** has only startup logs and no request logs when the gateway calls it, the gateway is likely **not reaching** sheets-mcp. **Fix:** In ECS **bridge** mode, use **container names** in the gateway config, not localhost: `host: http://redshift-mcp:9001/mcp` and `host: http://sheets-mcp:9002/mcp`. If the task def has `localhost:9001` / `localhost:9002`, the agent-gateway container tries to connect to itself; sheets-mcp never receives requests. Update the inline `agentgateway.yaml` in ag-config-init to use **redshift-mcp** and **sheets-mcp** hostnames, then redeploy. |
| **3. OPTIONS 405** | If the client sends CORS preflight (e.g. MCP Inspector in browser), ECS gateway must allow **OPTIONS** on `/mcp`. Agent Gateway (Go) with CORS config in `agentgateway.yaml` usually handles this. If ECS is still an older Python gateway build without the OPTIONS middleware, upgrade the ECS task to use the Agent Gateway image (same as local) or a Python image that includes OPTIONS handling. |
| **4. ALB / ECS health check → 406** | GET **/healthz** (or GET /mcp) is routed through the MCP handler, which returns **406** (`client must accept both application/json and text/event-stream`). So HTTP health checks that expect 200 fail. **Fix:** Use a **TCP** health check on port 80 (or 8080) at the **target group** (ALB). For the **container** health check in the task definition, use a port probe instead of GET /healthz (e.g. `command: ["CMD-SHELL", "timeout 2 bash -c '</dev/tcp/127.0.0.1/8080' || exit 1"]` if the image has bash, or keep wget only if your image treats 406 as success). |
| **5. OPTIONS /mcp → 405** | Some clients (e.g. browser) send **OPTIONS** for CORS preflight. Agent Gateway can return **405** (`method not allowed; must be GET, POST, or DELETE`). **Fix:** Ensure the gateway CORS config allows OPTIONS; or use a gateway version that responds 200/204 to OPTIONS on /mcp. If Cursor works (it uses POST directly), 405 may only affect MCP Inspector in a browser. |

**Temporary workaround:** Use local Docker and point `.mcp.json` at `http://localhost:8080/mcp` while ECS is fixed.

---

## Gateway logs (ECS / ops)

When debugging the **agent-gateway** (e.g. in CloudWatch), use this to interpret common log lines:

| Log message | Meaning | What to do |
|-------------|---------|------------|
| **http.status=406** — `client must accept both application/json and text/event-stream` | Request was **GET** (e.g. to `/healthz` or `/mcp`) or missing MCP `Accept` headers. The gateway routes these through the MCP handler, which requires POST with `Accept: application/json` and `Accept: text/event-stream`. | **Health checks:** Use **TCP** on port 80/8080 at the ALB target group instead of HTTP GET /healthz. For ECS container health, use a port probe or a command that does not require 200 from /healthz. **Clients:** Cursor/Claude Code send POST + headers; 406 usually comes from load balancer or GET probes. |
| **OPTIONS /mcp → 405** — `method not allowed; must be GET, POST, or DELETE` | CORS preflight (OPTIONS) is not allowed by the gateway for `/mcp`. | Ensure Agent Gateway CORS config responds to OPTIONS (200/204). If Cursor works over POST, 405 may only affect browser-based MCP Inspector. |
| **http.status=404** — `upstream error: 404 Not Found` | The gateway forwarded the request (e.g. `mcp.method=initialize`) to an **upstream** (redshift-mcp or sheets-mcp); that upstream returned 404. | **Redshift-mcp** often shows `POST /mcp 200 OK` in its own logs when it’s healthy. So 404 usually means the **other** upstream (e.g. **sheets-mcp**) is returning 404. Check sheets-mcp: is it running, and does it serve MCP at `POST /mcp`? Fix or redeploy the failing upstream. |
| **http.status=500** on `mcp.method=initialize` | Gateway called both redshift and sheets; one returned an error. Usually **sheets-mcp** (missing `GOOGLE_SA_KEY_BASE64` or `SPREADSHEET_ID`). | **Fix A:** Set Sheets credentials in `.env` and restart. **Fix B (Redshift only):** In `_backup/gateway/.env` set `GATEWAY_REDSHIFT_ONLY=1`, then `docker compose up -d` so 8080 works with Redshift only. |

**Quick check:** In the same time window, look for `redshift-mcp` and `sheets-mcp` container logs. Whichever upstream does **not** log `POST /mcp 200 OK` for the same session is the one returning 404.

#### MCP Inspector (`npx @modelcontextprotocol/inspector`) gets 404 when connecting

The inspector connects to the same gateway URL and sends **initialize** first. The gateway **federates** multiple backends (redshift + sheets): it forwards `initialize` to **all** of them and aggregates. If **any** upstream returns 404, the gateway returns 404 to the client. So from the inspector’s point of view it “never reaches the MCPs” — it fails at **registration** (initialize), before listing or calling tools.

| What you see | Cause | Fix |
|--------------|--------|-----|
| Inspector connects to gateway URL, then 404 | One of the gateway’s upstreams (redshift-mcp or sheets-mcp) returned 404 for `initialize`. | Same as above: find which upstream returns 404 (check redshift-mcp and sheets-mcp logs for the same request time). Usually **sheets-mcp** (wrong path, not running, or not serving `POST /mcp`). Fix or redeploy that upstream so **both** backends respond 200 to `POST /mcp`. |

**Note:** The request **does** reach the gateway and is forwarded; the failure is an upstream 404. Ensure both `redshift-mcp` and `sheets-mcp` are healthy and expose `POST /mcp`.

#### Local: POST /mcp returns 404 — find the failing upstream

1. From `_backup/gateway` with the stack up (`docker compose up -d`), run:
   ```bash
   chmod +x check-upstreams.sh && ./check-upstreams.sh
   ```
   You’ll see which of `redshift-mcp:9001/mcp` or `sheets-mcp:9002/mcp` returns 200 vs 404.

2. Check logs for the one that’s not 200:
   ```bash
   docker compose logs redshift-mcp
   docker compose logs sheets-mcp
   ```

3. **Sheets** often fails without credentials: set `GOOGLE_SA_KEY_BASE64` or `SERVICE_ACCOUNT_PATH` (and `SPREADSHEET_ID`) in `.env`. Restart: `docker compose up -d`.

4. **Redshift** needs `REDSHIFT_HOST`, `REDSHIFT_PORT`, `REDSHIFT_DB`, `REDSHIFT_USER`, `REDSHIFT_PASSWORD` in `.env`. If any are missing, the Redshift MCP may not start or may return errors.

5. After both backends return 200, connect again to `http://localhost:8080/mcp`.

#### Enabling Sheets (gateway at 8080 with both Redshift + Sheets)

1. Get a **Google Cloud service account** JSON key with Sheets API enabled.
2. In `_backup/gateway/.env`: set `GOOGLE_SA_KEY_BASE64=` to the base64 of that JSON (single line: `base64 -i key.json | tr -d '\n'`), and set `SPREADSHEET_ID=` to your ECM sheet ID.
3. Unset or set `GATEWAY_REDSHIFT_ONLY=0`, then `docker compose up -d`. Connect to `http://localhost:8080/mcp` for both Redshift and Sheets tools.

#### Fault isolation (one backend failing doesn’t break the other)

The single gateway at 8080 calls **both** backends for initialize; if one fails, the whole connection fails. To isolate:

- **8080** — Redshift-only (set `GATEWAY_REDSHIFT_ONLY=1` in `.env`) or full config.
- **8081** — Sheets-only gateway (same compose; runs `agent-gateway-sheets`).

Add **two** MCP servers: `http://localhost:8080/mcp` (ecm-redshift) and `http://localhost:8081/mcp` (ecm-sheets). If one backend is down, the other still works. See `_backup/gateway/README.md` for details.

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | Open repo/workspace containing `work-plugins/ecm-operations` |
| 2 | Choose ECS (VPN) or Local Docker; set gateway URL in `.mcp.json` or MCP config |
| 3 | Confirm ecm-gateway is the only Redshift/Sheets MCP |
| 4 | Trust the plugin if prompted |
| 5 | Reload window after any MCP change |
| 6 | Test with **"Run ECM"** or **"order &lt;id&gt;"** |

After this, agents can use **"run ECM"**, **"order &lt;id&gt;"**, **"my tickets"**, **"resolve"**, **"escalate"** from their local machine with Claude and ecm-gateway.
