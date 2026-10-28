"""Controlled probe: which input (images vs proprio) drives the large first-step jump on live robot requests."""
import json, sys, glob
import numpy as np, requests, json_numpy

json_numpy.patch()
np.set_printoptions(suppress=True, linewidth=200, precision=1)
ROOT = "/home/yhkim/openvla-oft/outputs"
base = json.load(open(f"{ROOT}/so101-server-validation-20260928-1407/validated/sample-request.json"))
full = json_numpy.loads(json.dumps(base["full_image"]))
add = [json_numpy.loads(json.dumps(x)) for x in base["additional_images"]]
st0 = np.asarray(json_numpy.loads(json.dumps(base["state"])), dtype=np.float32)
instr = base["instruction"] or "Pick up the block and place it into the cup."

# Unique live robot states from today's bidir runs
live = []
for f in [f"{ROOT}/bidir-15000-20261012/actions.jsonl", f"{ROOT}/bidir-7500-20261012/actions.jsonl"]:
    for l in open(f):
        r = json.loads(l)
        if "startup" in r["request_id"]:
            continue
        s = np.round(np.array(r["state"]), 1)
        if not any(np.allclose(s, x) for x, _ in live):
            live.append((s, np.array(r["actions"][0]) - np.array(r["state"])))
live = live[:6]


def ask(port, full_img, add_imgs, state, tag):
    obs = {"full_image": full_img, "additional_images": add_imgs, "state": np.asarray(state, np.float32),
           "instruction": instr, "request_id": f"claude-probe-{tag}"}
    r = requests.post(f"http://127.0.0.1:{port}/act", data=json_numpy.dumps(obs),
                      headers={"Content-Type": "application/json"}, timeout=120)
    r.raise_for_status()
    a = np.asarray(json_numpy.loads(r.text) if isinstance(r.json(), str) else r.json())
    return a


for port in map(int, sys.argv[1:]):
    print(f"===== port {port}")
    a = ask(port, full, add, st0, "A")
    print("A dataset imgs + dataset state   d0", a[0] - st0, "rng", np.ptp(a, 0))
    black = np.zeros_like(full)
    a = ask(port, black, [black, black], st0, "C")
    print("C black imgs  + dataset state   d0", a[0] - st0, "rng", np.ptp(a, 0))
    a = ask(port, add[0], [full, add[1]], st0, "D")
    print("D front<->top swap + dataset st d0", a[0] - st0, "rng", np.ptp(a, 0))
    a = ask(port, full[..., ::-1].copy(), [x[..., ::-1].copy() for x in add], st0, "E")
    print("E RGB->BGR    + dataset state   d0", a[0] - st0, "rng", np.ptp(a, 0))
    for i, (s, livedelta) in enumerate(live):
        a = ask(port, full, add, s, f"B{i}")
        print(f"B{i} dataset imgs + live state {s}\n    d0", a[0] - s, " | live-imgs d0 was", np.round(livedelta, 1))
