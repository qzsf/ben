import simpy
import random
import numpy as np
from collections import defaultdict

# -----------------------------
# Helpers: distributions & stats
# -----------------------------
def lognormal_from_mean_cv(mean, cv, rng):
    """
    Sample a lognormal with a given mean and coefficient of variation (cv = std/mean).
    Good for service times (positive + right-skewed).
    """
    if mean <= 0:
        return 0.0
    if cv <= 0:
        return float(mean)

    sigma2 = np.log(1 + cv**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return float(rng.lognormal(mean=mu, sigma=sigma))

def percentile(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(np.array(arr, dtype=float), p))

def summarize_waits(waits):
    if len(waits) == 0:
        return {"n": 0}
    a = np.array(waits, dtype=float)
    return {
        "n": int(len(a)),
        "avg": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }

# -----------------------------
# Service catalog (example)
# -----------------------------
SERVICE_TYPES = {
    # name: (mean_minutes, cv)
    "renewal_simple": (6.0, 0.6),
    "written_test":   (12.0, 0.7),
    "real_id":        (18.0, 0.8),
    "title_transfer": (22.0, 0.9),
}

# Mix by customer class (example probabilities; must sum to 1)
MIX = {
    "appt": {
        "renewal_simple": 0.35,
        "written_test":   0.20,
        "real_id":        0.30,
        "title_transfer": 0.15,
    },
    "walkin": {
        "renewal_simple": 0.45,
        "written_test":   0.25,
        "real_id":        0.20,
        "title_transfer": 0.10,
    }
}

def choose_service_type(kind, rng):
    items = list(MIX[kind].items())
    names = [k for k, _ in items]
    probs = [v for _, v in items]
    return rng.choice(names, p=probs)

def sample_service_time(service_type, rng):
    mean, cv = SERVICE_TYPES[service_type]
    return lognormal_from_mean_cv(mean, cv, rng)

# -----------------------------
# Customer with abandonment
# -----------------------------
def customer(env, kind, cid, clerks, priority, max_wait, rng, metrics):
    """
    Customer arrives, requests a clerk, abandons if waiting > max_wait.
    Records wait times, abandonment, and service type durations.
    """
    arrive = env.now
    service_type = choose_service_type(kind, rng)

    req = clerks.request(priority=priority)

    # Two competing events:
    # 1) request granted
    # 2) abandonment timeout hits first
    results = yield req | env.timeout(max_wait)

    if req not in results:
        # Abandoned before getting served
        req.cancel()  # important: remove from resource queue
        metrics["abandoned"][kind] += 1
        metrics["abandon_waits"][kind].append(env.now - arrive)
        metrics["by_service_abandoned"][service_type] += 1
        return

    # Served
    start = env.now
    wait = start - arrive
    metrics["waits"][kind].append(wait)
    metrics["served"][kind] += 1
    metrics["by_service_served"][service_type] += 1
    metrics["service_type_waits"][(kind, service_type)].append(wait)

    # Service time
    st = sample_service_time(service_type, rng)
    metrics["service_times"][(kind, service_type)].append(st)
    yield env.timeout(st)

    clerks.release(req)

# -----------------------------
# Appointment schedule: 30-min slots
# -----------------------------
def appointment_scheduler(
    env,
    clerks,
    priority,
    rng,
    metrics,
    sim_minutes,
    slot_minutes=30,
    appts_per_slot=25,
    no_show_rate=0.10,
    arrival_jitter_std=3.0,  # minutes (small randomness around slot start)
    max_wait=30
):
    """
    Every slot_minutes, generate appts_per_slot appointment customers.
    They "arrive" around the slot start with small jitter.
    """
    cid = 0
    t = 0
    while t < sim_minutes:
        slot_start = t

        # Create appointment arrivals for this slot
        for _ in range(appts_per_slot):
            # no-show
            if rng.random() < no_show_rate:
                metrics["no_shows"] += 1
                continue

            # arrival time around slot start
            jitter = rng.normal(0, arrival_jitter_std)
            arrive_time = max(slot_start, slot_start + jitter)  # don't arrive before slot start too much
            cid += 1

            def delayed_start(arrive_at, _cid):
                yield env.timeout(arrive_at - env.now)
                env.process(customer(
                    env, "appt", _cid, clerks, priority,
                    max_wait=max_wait, rng=rng, metrics=metrics
                ))

            env.process(delayed_start(arrive_time, cid))

        # advance to next slot
        t += slot_minutes
        yield env.timeout(slot_minutes)

# -----------------------------
# Walk-in arrivals: time-varying (piecewise rates)
# -----------------------------
def walkin_arrivals(
    env,
    clerks,
    priority,
    rng,
    metrics,
    sim_minutes,
    max_wait=45,
    # piecewise arrival rates per hour: [(start_min, end_min, rate_per_hour), ...]
    walkin_rate_schedule=None
):
    """
    Walk-ins arrive as a Poisson process with time-varying rate.
    """
    if walkin_rate_schedule is None:
        # Default: slower early, peak midday, slower late
        walkin_rate_schedule = [
            (0,         120,  25),  # first 2 hours
            (120,       360,  45),  # mid 4 hours peak
            (360, sim_minutes, 30),  # last period
        ]

    def rate_per_min_at(tmin):
        for s, e, rph in walkin_rate_schedule:
            if s <= tmin < e:
                return rph / 60.0
        return walkin_rate_schedule[-1][2] / 60.0

    cid = 0
    while env.now < sim_minutes:
        lam = rate_per_min_at(env.now)
        if lam <= 0:
            # no arrivals in this interval; jump a bit
            yield env.timeout(1)
            continue

        interarrival = random.expovariate(lam)
        yield env.timeout(interarrival)

        if env.now >= sim_minutes:
            break

        cid += 1
        env.process(customer(
            env, "walkin", cid, clerks, priority,
            max_wait=max_wait, rng=rng, metrics=metrics
        ))

# -----------------------------
# Runner
# -----------------------------
def run_dmv_sim_realistic(
    seed=7,
    sim_minutes=8*60,
    num_clerks=8,

    # priorities (lower number => higher priority)
    appt_priority=0,
    walkin_priority=1,

    # abandonment thresholds (minutes)
    appt_max_wait=20,
    walkin_max_wait=45,

    # appointment scheduling
    slot_minutes=30,
    appts_per_slot=25,
    no_show_rate=0.10,
    arrival_jitter_std=3.0,

    # walk-in policy shock
    walkin_multiplier=1.20
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    env = simpy.Environment()
    clerks = simpy.PriorityResource(env, capacity=num_clerks)

    metrics = {
        "waits": defaultdict(list),
        "abandon_waits": defaultdict(list),
        "served": defaultdict(int),
        "abandoned": defaultdict(int),
        "no_shows": 0,
        "service_type_waits": defaultdict(list),
        "service_times": defaultdict(list),
        "by_service_served": defaultdict(int),
        "by_service_abandoned": defaultdict(int),
    }

    # Appointment process (slot-based)
    env.process(appointment_scheduler(
        env=env,
        clerks=clerks,
        priority=appt_priority,
        rng=rng,
        metrics=metrics,
        sim_minutes=sim_minutes,
        slot_minutes=slot_minutes,
        appts_per_slot=appts_per_slot,
        no_show_rate=no_show_rate,
        arrival_jitter_std=arrival_jitter_std,
        max_wait=appt_max_wait
    ))

    # Walk-in arrivals (time-varying Poisson). Apply multiplier to schedule rates.
    base_schedule = [
        (0,         120,  25),
        (120,       360,  45),
        (360, sim_minutes, 30),
    ]
    scaled_schedule = [(s, e, rph * walkin_multiplier) for (s, e, rph) in base_schedule]

    env.process(walkin_arrivals(
        env=env,
        clerks=clerks,
        priority=walkin_priority,
        rng=rng,
        metrics=metrics,
        sim_minutes=sim_minutes,
        max_wait=walkin_max_wait,
        walkin_rate_schedule=scaled_schedule
    ))

    env.run(until=sim_minutes)

    # Summaries
    out = {
        "served_appt": metrics["served"]["appt"],
        "served_walkin": metrics["served"]["walkin"],
        "abandoned_appt": metrics["abandoned"]["appt"],
        "abandoned_walkin": metrics["abandoned"]["walkin"],
        "no_shows": metrics["no_shows"],
        "appt_wait_summary": summarize_waits(metrics["waits"]["appt"]),
        "walkin_wait_summary": summarize_waits(metrics["waits"]["walkin"]),
        "appt_abandon_wait_summary": summarize_waits(metrics["abandon_waits"]["appt"]),
        "walkin_abandon_wait_summary": summarize_waits(metrics["abandon_waits"]["walkin"]),
        "by_service_served": dict(metrics["by_service_served"]),
        "by_service_abandoned": dict(metrics["by_service_abandoned"]),
    }

    return out

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    base = run_dmv_sim_realistic(walkin_multiplier=1.0)
    shock = run_dmv_sim_realistic(walkin_multiplier=1.2)

    print("BASELINE")
    print(base["appt_wait_summary"], base["walkin_wait_summary"])
    print("abandoned:", base["abandoned_appt"], base["abandoned_walkin"], "no_shows:", base["no_shows"])
    print("POLICY SHOCK (+20% walk-ins)")
    print(shock["appt_wait_summary"], shock["walkin_wait_summary"])
    print("abandoned:", shock["abandoned_appt"], shock["abandoned_walkin"], "no_shows:", shock["no_shows"])
