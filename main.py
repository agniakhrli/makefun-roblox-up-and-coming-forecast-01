import os
import json
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict

def parse_iso(dt_str):
    if not dt_str:
        return None
    # Handle 'Z' and fractional seconds
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, 'task.json'), 'r', encoding='utf-8') as f:
        task = json.load(f)

    history_path = os.path.join(input_dir, 'history.ndjson')
    reads = []
    with open(history_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))

    # Thin history to at most 200 reads (deterministic stride, always keeping the last read)
    if len(reads) > 200:
        step = math.ceil(len(reads) / 200)
        thinned_indices = [i for i in range(len(reads)) if i % step == 0 or i == len(reads) - 1]
        reads = [reads[i] for i in thinned_indices]

    # Parse read timestamps and up_and_coming charts
    parsed_reads = []
    first_seen = {}
    game_created = {}

    for idx, r in enumerate(reads):
        at_dt = parse_iso(r.get('at'))
        up_list = r.get('up_and_coming', []) or []
        details = r.get('details') or {}
        
        ranks = {}
        for item in up_list:
            uid = item.get('universe_id')
            rank = item.get('rank')
            if uid is not None and rank is not None:
                ranks[uid] = rank
                if uid not in first_seen and at_dt:
                    first_seen[uid] = at_dt

        # Check details for created time
        # details can be a dict mapping uid (str or int) to detail dict
        if isinstance(details, dict):
            for k, d in details.items():
                if isinstance(d, dict):
                    try:
                        uid_int = int(k)
                    except ValueError:
                        uid_int = None
                    created_str = d.get('created')
                    if created_str and at_dt:
                        cdt = parse_iso(created_str)
                        if cdt:
                            if uid_int is not None:
                                game_created[uid_int] = cdt
                            # also check items in up_list matching uid
        for item in up_list:
            uid = item.get('universe_id')
            # sometimes details is keyed differently or item itself has created
            if uid and uid not in game_created:
                # check if details has uid as key
                d = details.get(str(uid)) or details.get(uid)
                if isinstance(d, dict) and d.get('created'):
                    cdt = parse_iso(d.get('created'))
                    if cdt:
                        game_created[uid] = cdt

        parsed_reads.append({
            'at': at_dt,
            'ranks': ranks,
            'snapshot': r
        })

    # Determine expiry for each game: created (else first seen) + 28 days
    expiry = {}
    latest_at = parsed_reads[-1]['at'] if parsed_reads else datetime.now(timezone.utc)
    for uid, fs in first_seen.items():
        c_dt = game_created.get(uid, fs)
        expiry[uid] = c_dt + timedelta(days=28)

    # Horizon grid H
    H = [0, 2, 4, 6, 9, 12, 18, 24, 36, 48, 60, 72]
    
    # Rank buckets: exact ranks 1..10, then coarser centres 13, 18, 25, 40, 70, 97
    centers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 18, 25, 40, 70, 97]

    def get_bucket_weights(r):
        if r <= 1:
            return {1: 1.0}
        if r >= 97:
            return {97: 1.0}
        # Find interval
        for k in range(len(centers) - 1):
            c1 = centers[k]
            c2 = centers[k+1]
            if c1 <= r <= c2:
                if c1 == c2:
                    return {c1: 1.0}
                w2 = (r - c1) / (c2 - c1)
                w1 = 1.0 - w2
                res = {c1: w1}
                if w2 > 0:
                    res[c2] = w2
                return res
        return {97: 1.0}

    # Outcomes: 0..9 for future place 1..10, 10 for rank 11..97, 11 for left chart
    num_outcomes = 12

    # counts[h_idx][bucket][outcome]
    # Initialize transition counts with Laplace prior of 1.0 later, or accumulate raw counts first.
    # To accumulate: counts[h][b][o]
    raw_counts = {}
    for h in H:
        raw_counts[h] = {b: [0.0] * num_outcomes for b in centers}

    # Loop over pairs i < j
    n_reads = len(parsed_reads)
    for i in range(n_reads):
        ri = parsed_reads[i]
        ti = ri['at']
        if not ti:
            continue
        ranks_i = ri['ranks']

        for j in range(i + 1, n_reads):
            rj = parsed_reads[j]
            tj = rj['at']
            if not tj:
                continue
            dt = (tj - ti).total_seconds() / 3600.0
            if dt < 0 or dt > 84:
                continue

            # Recency weight on later read j (half-life 168 hours relative to latest_at)
            # w_rec = 2 ** (- (latest_at - tj).total_seconds() / (168 * 3600))
            # Or relative to ti or latest_at? Rule 3: "a recency weight on the later read with half-life 168 hours"
            # Let's use 2 ** (- (latest_at - tj).total_seconds() / (168 * 3600))
            age_j_hours = (latest_at - tj).total_seconds() / 3600.0
            w_rec = 2.0 ** (-max(0.0, age_j_hours) / 144.0)

            ranks_j = rj['ranks']

            # For each game in ranks_i
            for uid, rank_val in ranks_i.items():
                exp_dt = expiry.get(uid, ti + timedelta(days=28))
                
                b_weights = get_bucket_weights(rank_val)

                # Determine future outcome at j
                if uid not in ranks_j:
                    outcome = 11 # left chart
                else:
                    fut_rank = ranks_j[uid]
                    if 1 <= fut_rank <= 10:
                        outcome = fut_rank - 1
                    else:
                        outcome = 10 # rank 11..97

                for h_idx, h in enumerate(H):
                    # Check age-out: if expiry is before ti + h, skip
                    if ti + timedelta(hours=h) > exp_dt:
                        continue

                    gauss = math.exp(-((dt - h) ** 2) / (2.0 * (5.5 ** 2)))
                    if gauss < 0.01:
                        continue

                    weight = gauss * w_rec
                    for b, bw in b_weights.items():
                        raw_counts[h][b][outcome] += bw * weight

    # Add Laplace prior of 1.0 to every outcome count, then normalize
    transition_probs = {}
    for h in H:
        transition_probs[h] = {}
        for b in centers:
            counts = [c + 1.0 for c in raw_counts[h][b]]
            total = sum(counts)
            transition_probs[h][b] = [c / total for c in counts]

    # Trapezoid weights over horizon grid H
    # H = [0, 2, 4, 6, 9, 12, 18, 24, 36, 48, 60, 72]
    trap_widths = []
    n_h = len(H)
    for k in range(n_h):
        if k == 0:
            w = (H[1] - H[0]) / 2.0
        elif k == n_h - 1:
            w = (H[k] - H[k-1]) / 2.0
        else:
            w = (H[k+1] - H[k-1]) / 2.0
        trap_widths.append(w)
    total_w = sum(trap_widths)
    trap_weights = [w / total_w for w in trap_widths]

    # Last read chart for baseline / horizon 0 / active games
    last_read = parsed_reads[-1]
    last_ranks = last_read['ranks']
    # Sort chart by rank
    sorted_chart_uids = sorted(last_ranks.keys(), key=lambda uid: last_ranks[uid])

    # Compute expected probabilities for each game currently on chart (or active)
    # We evaluate all games in last_ranks (and any active game)
    game_probs = {} # uid -> {place: prob}

    active_uids = set(last_ranks.keys())
    # Also add any game seen in recent reads if needed, but last read chart games are primary
    for uid in active_uids:
        rank_val = last_ranks[uid]
        b_weights = get_bucket_weights(rank_val)

        # For each place 1..10 (outcome 0..9)
        place_probs = {p: 0.0 for p in range(1, 11)}

        for p in range(1, 11):
            outcome = p - 1
            # Average over horizon grid using trap_weights
            p_val = 0.0
            for k, h in enumerate(H):
                # At horizon 0, rule 7: "At horizon 0 the chart is exactly the last read (the game at rank k has probability 1 at place k)."
                if h == 0:
                    # If game was at rank k in last read, and place p == k, prob is 1.0, else 0.0
                    h_prob = 1.0 if rank_val == p else 0.0
                else:
                    # Sum over bucket weights
                    h_prob = 0.0
                    for b, bw in b_weights.items():
                        h_prob += bw * transition_probs[h][b][outcome]
                
                p_val += trap_weights[k] * h_prob
            
            place_probs[p] = max(0.0, p_val)

        game_probs[uid] = place_probs

    # Assemble rows for probabilities (Rule 10)
    # "for each place 1..10, take the games with the highest probability at that place (probability above 0.0005),
    # at most 18 per place, and renormalise each place's named mass to 0.75 (so 0.25 is left unnamed for the long tail).
    # Round each p to 6 decimals. Sort rows by place, then descending p, then universe_id."
    probabilities_rows = []

    for p in range(1, 11):
        place_candidates = []
        for uid, p_dict in game_probs.items():
            val = p_dict.get(p, 0.0)
            if val > 0.0005:
                place_candidates.append((val, uid))
        
        # Sort descending by p, then ascending by universe_id
        place_candidates.sort(key=lambda x: (-x[0], x[1]))
        place_candidates = place_candidates[:18]

        if place_candidates:
            total_mass = sum(v for v, uid in place_candidates)
            if total_mass > 0:
                scale = 0.75 / total_mass
                for val, uid in place_candidates:
                    p_rounded = round(val * scale, 6)
                    probabilities_rows.append({
                        "universe_id": int(uid),
                        "place": int(p),
                        "p": float(p_rounded)
                    })

    # Sort rows by place, then descending p, then universe_id
    probabilities_rows.sort(key=lambda r: (r['place'], -r['p'], r['universe_id']))

    # Top10 (Rule 11):
    # "order the games by expected place (sum over places of place * p(place), divided by the game's total probability over the 10 places),
    # lowest first, and take the first 10."
    game_scores = []
    for uid in active_uids:
        p_dict = game_probs.get(uid, {p: 0.0 for p in range(1, 11)})
        total_p = sum(p_dict.values())
        if total_p > 0:
            weighted_sum = sum(p * p_dict[p] for p in range(1, 11))
            expected_place = weighted_sum / total_p
        else:
            expected_place = 999.0
        game_scores.append((expected_place, uid))

    # Sort by expected place ascending, then universe_id ascending
    game_scores.sort(key=lambda x: (x[0], x[1]))
    top10 = [int(uid) for _, uid in game_scores[:10]]

    # Ensure top10 has exactly 10 distinct universe ids (fallback to sorted_chart_uids if needed)
    if len(top10) < 10:
        for uid in sorted_chart_uids:
            if uid not in top10:
                top10.append(int(uid))
                if len(top10) == 10:
                    break

    output_data = {
        "probabilities": probabilities_rows,
        "top10": top10
    }

    out_file = os.path.join(output_dir, task.get('output', 'forecast.json'))
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, separators=(',', ':'))

if __name__ == '__main__':
    main()
